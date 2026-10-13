import os
import logging
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("summarizer")

INPUT_FILE = "/data/ingested.txt"
OUTPUT_FILE = "/data/summary.txt"

SYSTEM_PROMPT = (
    "You are a helpful assistant that summarizes long text "
    "into key bullet points. Each bullet should be one "
    "concise sentence capturing a core insight."
)

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

OLLAMA_URL = "https://ollama.com/api/chat"
MODEL = "gpt-oss:120b-cloud"


def summarize(text):
    response = requests.post(
        OLLAMA_URL,
        headers={
            "Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text[:8000]},
            ],
            "stream": False,
        },
        timeout=180,
    )

    response.raise_for_status()
    return response.json()["message"]["content"]

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()

    if not raw_text.strip():
        logger.warning("Empty input. Writing fallback summary.")
        summary = "No content to summarize."
    else:
        try:
            summary = summarize(raw_text)
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            summary = f"Summarization failed: {e}"

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(summary)
    logger.info(f"Summary written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()