from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
import dotenv

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

def get_llm(model: str | None = None) -> ChatOpenAI:
    model = model or os.getenv(
        "AEGIS_MODEL",
        "openai/gpt-oss-20b"
    )

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set."
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=GROQ_BASE_URL,
        temperature=0.1,
    )