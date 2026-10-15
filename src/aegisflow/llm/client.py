import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI


DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"


def get_llm(model: str | None = None) -> ChatOpenAI:
    model = model or os.getenv("AEGIS_MODEL", DEFAULT_MODEL)

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set."
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=os.getenv("AEGIS_BASE_URL", DEFAULT_BASE_URL),
        temperature=0.1,
    )