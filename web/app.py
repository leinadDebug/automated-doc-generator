# web/app.py
import os
import tempfile
import shutil
import json
import uuid
from pathlib import Path
from flask import (
    Flask,
    render_template,
    request,
    send_file,
    session,
    redirect,
    url_for,
    after_this_request,
)
from werkzeug.utils import secure_filename

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import (
    from_paste,
    from_file,
    from_folder,
    from_github,
    combine_files,
    detect_language_from_extension,
)
from src.prompt import build_prompt
from src.ai_client import generate_docs
from src.markdown_utils import render_markdown_to_html, markdown_to_body_html
from src.cli import markdown_to_json

app = Flask(__name__)
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY", "dev-secret-key-change-in-production"
)

DOCS_DIR = os.path.join(tempfile.gettempdir(), "adoc_generated")
os.makedirs(DOCS_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".java",
    ".c",
    ".cpp",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".cs",
    ".sh",
    ".pl",
    ".pm",
    ".lua",
    ".r",
    ".m",
    ".groovy",
    ".dart",
    ".jl",
    ".ex",
    ".exs",
}


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    title = request.form.get("title", "")
    author = request.form.get("author", "")
    version = request.form.get("version", "")
    context = request.form.get("context", "")
    language = request.form.get("language", "")
    format_choice = request.form.get("format", "md")

    provider = request.form.get("provider", "deepseek")

    if provider == "deepseek":
        api_key = request.form.get("deepseek_api_key", "").strip() or None
    elif provider == "gemini":
        api_key = request.form.get("gemini_api_key", "").strip() or None
    else:
        return render_template("index.html", error="Invalid AI provider selected.")

    if not api_key:
        return render_template("index.html", error="Please enter an API key.")

    input_type = request.form.get("input_type")
    files = []

    # Input handling
    if input_type == "paste":
        code = request.form.get("code", "")
        if not code:
            return render_template("index.html", error="Please paste some code.")
        files = from_paste(code)

    elif input_type == "file":
        uploaded_files = request.files.getlist("files")
        if not uploaded_files:
            return render_template(
                "index.html", error="Please select at least one file."
            )
        temp_dir = tempfile.mkdtemp(prefix="adoc_upload_")
        for f in uploaded_files:
            if f.filename and allowed_file(f.filename):
                filename = secure_filename(f.filename)
                filepath = os.path.join(temp_dir, filename)
                f.save(filepath)
        files = from_folder(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    elif input_type == "github":
        github_url = request.form.get("github_url", "")
        branch = request.form.get("branch", "")
        if not github_url:
            return render_template("index.html", error="Please enter a GitHub URL.")
        try:
            files = from_github(github_url, branch=branch if branch else None)
        except Exception as e:
            return render_template("index.html", error=f"GitHub clone failed: {str(e)}")

    else:
        return render_template("index.html", error="Invalid input type.")

    if not files:
        return render_template("index.html", error="No code files found.")

    combined_code = combine_files(files)

    if not language:
        lang = detect_language_from_extension(files[0][0])
        if lang == "Unknown":
            lang = "Python"
        language = lang

    prompt = build_prompt(
        language=language,
        project_name=title,
        author=author,
        version=version,
        context=context,
    )

    # Generate documentation with optional API key
    try:
        doc = generate_docs(prompt, combined_code, api_key=api_key, provider=provider)
    except Exception as e:
        return render_template("index.html", error=f"AI client error: {str(e)}")

    if doc is None:
        return render_template(
            "index.html",
            error="Documentation generation failed. Please check your API key and try again.",
        )

    # Store in temporary file
    doc_id = str(uuid.uuid4())
    doc_path = os.path.join(DOCS_DIR, f"{doc_id}.txt")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(doc)

    session["doc_id"] = doc_id
    session["title"] = title or "Documentation"
    session["format"] = format_choice

    return redirect(url_for("result"))


@app.route("/result")
def result():
    doc_id = session.get("doc_id")
    title = session.get("title", "Documentation")
    format_choice = session.get("format", "md")

    if not doc_id:
        return redirect(url_for("index"))

    doc_path = os.path.join(DOCS_DIR, f"{doc_id}.txt")
    if not os.path.exists(doc_path):
        return redirect(url_for("index"))

    with open(doc_path, "r", encoding="utf-8") as f:
        doc = f.read()

    html_preview = markdown_to_body_html(doc)

    return render_template(
        "result.html", title=title, doc=html_preview, format_choice=format_choice
    )


@app.route("/download/<format>")
def download(format):
    doc_id = session.get("doc_id")
    title = session.get("title", "documentation")

    if not doc_id:
        return redirect(url_for("index"))

    doc_path = os.path.join(DOCS_DIR, f"{doc_id}.txt")
    if not os.path.exists(doc_path):
        return redirect(url_for("index"))

    with open(doc_path, "r", encoding="utf-8") as f:
        doc = f.read()

    if format == "md":
        content = doc
        ext = ".md"
        mimetype = "text/markdown"
    elif format == "html":
        content = render_markdown_to_html(doc, title)
        ext = ".html"
        mimetype = "text/html"
    elif format == "json":
        data = markdown_to_json(doc)
        content = json.dumps(data, indent=2, ensure_ascii=False)
        ext = ".json"
        mimetype = "application/json"
    else:
        return "Invalid format", 400

    download_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    download_file.write(content.encode("utf-8"))
    download_file.close()

    @after_this_request
    def cleanup(response):
        try:
            os.unlink(download_file.name)
        except:
            pass
        return response

    safe_title = secure_filename(title) or "documentation"
    return send_file(
        download_file.name,
        as_attachment=True,
        download_name=f"{safe_title}{ext}",
        mimetype=mimetype,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
