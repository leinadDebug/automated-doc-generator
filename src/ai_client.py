import time
import logging
import requests
from typing import Optional

from src.config import DEEPSEEK_API_KEY, GEMINI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepSeekClient:
    def __init__(
        self, api_key: str = DEEPSEEK_API_KEY, model: str = "deepseek-v4-flash"
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.max_retries = 3
        self.retry_delay = 2

    def generate_documentation(
        self, system_prompt: str, user_code: str, max_tokens: int = 300000
    ) -> Optional[str]:

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_code},
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "top_p": 0.9,
        }

        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Sending request to DeepSeek "
                    f"(attempt {attempt + 1}/{self.max_retries})..."
                )

                response = requests.post(
                    self.base_url, headers=headers, json=payload, timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]

                    logger.info("Documentation generated successfully with DeepSeek.")

                    return content

                elif response.status_code == 429:
                    wait_time = self.retry_delay * (2**attempt)

                    logger.warning(
                        f"DeepSeek rate limited (429). " f"Retrying in {wait_time}s..."
                    )

                    time.sleep(wait_time)

                elif response.status_code in (500, 502, 503, 504):
                    wait_time = self.retry_delay * (2**attempt)

                    logger.warning(
                        f"DeepSeek server error {response.status_code}. "
                        f"Retrying in {wait_time}s..."
                    )

                    time.sleep(wait_time)

                else:
                    logger.error(
                        f"DeepSeek API error {response.status_code}: "
                        f"{response.text}"
                    )

                    return None

            except requests.exceptions.Timeout:
                logger.warning(
                    f"DeepSeek request timed out "
                    f"(attempt {attempt + 1}). Retrying..."
                )

                time.sleep(self.retry_delay * (2**attempt))

            except requests.exceptions.ConnectionError:
                logger.warning(
                    f"DeepSeek connection error "
                    f"(attempt {attempt + 1}). Retrying..."
                )

                time.sleep(self.retry_delay * (2**attempt))

        logger.error("DeepSeek retries exhausted.")

        return None


class GeminiClient:
    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = "gemini-3.6-flash"):
        self.api_key = api_key
        self.model = model

        self.base_url = (
            f"https://generativelanguage.googleapis.com/"
            f"v1beta/models/{self.model}:generateContent"
        )

        self.max_retries = 3
        self.retry_delay = 2

    def generate_documentation(
        self, system_prompt: str, user_code: str, max_tokens: int = 300000
    ) -> Optional[str]:

        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

        # Gemini has its own output-token limits, so don't blindly
        # send the DeepSeek value of 300000.
        gemini_max_tokens = min(max_tokens, 8192)

        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_code}]}],
            "generationConfig": {
                "temperature": 0.3,
                "topP": 0.9,
                "maxOutputTokens": gemini_max_tokens,
            },
        }

        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Sending request to Gemini "
                    f"(attempt {attempt + 1}/{self.max_retries})..."
                )

                response = requests.post(
                    self.base_url, headers=headers, json=payload, timeout=60
                )

                if response.status_code == 200:
                    result = response.json()

                    candidates = result.get("candidates", [])

                    if not candidates:
                        logger.error("Gemini returned no candidates.")
                        return None

                    parts = candidates[0].get("content", {}).get("parts", [])

                    content = "".join(part.get("text", "") for part in parts)

                    if not content:
                        logger.error("Gemini returned an empty response.")
                        return None

                    logger.info("Documentation generated successfully with Gemini.")

                    return content

                elif response.status_code == 429:
                    wait_time = self.retry_delay * (2**attempt)

                    logger.warning(
                        f"Gemini rate limited (429). " f"Retrying in {wait_time}s..."
                    )

                    time.sleep(wait_time)

                elif response.status_code in (500, 502, 503, 504):
                    wait_time = self.retry_delay * (2**attempt)

                    logger.warning(
                        f"Gemini server error {response.status_code}. "
                        f"Retrying in {wait_time}s..."
                    )

                    time.sleep(wait_time)

                else:
                    logger.error(
                        f"Gemini API error {response.status_code}: " f"{response.text}"
                    )

                    return None

            except requests.exceptions.Timeout:
                logger.warning(
                    f"Gemini request timed out " f"(attempt {attempt + 1}). Retrying..."
                )

                time.sleep(self.retry_delay * (2**attempt))

            except requests.exceptions.ConnectionError:
                logger.warning(
                    f"Gemini connection error " f"(attempt {attempt + 1}). Retrying..."
                )

                time.sleep(self.retry_delay * (2**attempt))

        logger.error("Gemini retries exhausted.")

        return None


# ---------------------------------------------------------
# Clients
# ---------------------------------------------------------

_deepseek_client = None
_gemini_client = None


def get_deepseek_client(api_key: Optional[str] = None) -> DeepSeekClient:

    global _deepseek_client

    if api_key is not None:
        return DeepSeekClient(api_key=api_key)

    if _deepseek_client is None:
        _deepseek_client = DeepSeekClient()

    return _deepseek_client


def get_gemini_client(api_key: Optional[str] = None) -> GeminiClient:

    global _gemini_client

    if api_key is not None:
        return GeminiClient(api_key=api_key)

    if _gemini_client is None:
        _gemini_client = GeminiClient()

    return _gemini_client


# ---------------------------------------------------------
# Main generation function with fallback
# ---------------------------------------------------------


def generate_docs(
    system_prompt: str,
    user_code: str,
    api_key: Optional[str] = None,
    provider: str = "deepseek",
) -> Optional[str]:

    if provider == "deepseek":
        client = get_deepseek_client(api_key=api_key)

    elif provider == "gemini":
        client = get_gemini_client(api_key=api_key)

    else:
        logger.error(f"Unsupported AI provider: {provider}")
        return None

    return client.generate_documentation(system_prompt, user_code)
