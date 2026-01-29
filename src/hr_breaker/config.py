import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


def setup_logging() -> logging.Logger:
    general_level = os.getenv("LOG_LEVEL_GENERAL", "WARNING").upper()
    project_level = os.getenv("LOG_LEVEL", "WARNING").upper()

    logging.basicConfig(
        level=getattr(logging, general_level, logging.WARNING),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    project_logger = logging.getLogger("hr_breaker")
    project_logger.setLevel(getattr(logging, project_level, logging.WARNING))
    return project_logger


logger = setup_logging()


class Settings(BaseModel):
    """Application settings."""

    openai_api_key: str = ""
    openai_base_url: str = "https://openrouter.ai/api/v1"
    pro_model: str = "google/gemini-3-pro-preview"
    flash_model: str = "google/gemini-3-flash-preview"
    cache_dir: Path = Path(".cache/resumes")
    output_dir: Path = Path("output")
    max_iterations: int = 5
    pass_threshold: float = 0.7
    fast_mode: bool = True

    # Scraper settings
    scraper_httpx_timeout: float = 15.0
    scraper_wayback_timeout: float = 10.0
    scraper_playwright_timeout: int = 30000
    scraper_httpx_max_retries: int = 3
    scraper_wayback_max_age_days: int = 30
    scraper_min_text_length: int = 200

    # Filter thresholds
    filter_hallucination_threshold: float = 0.9
    filter_keyword_threshold: float = 0.25
    filter_llm_threshold: float = 0.7
    filter_vector_threshold: float = 0.4
    filter_ai_generated_threshold: float = 0.4

    # Resume length limits
    resume_max_chars: int = 4500
    resume_max_words: int = 520
    resume_page2_overflow_chars: int = 1000

    # Keyword matcher params
    keyword_tfidf_max_features: int = 200
    keyword_tfidf_cutoff: float = 0.1
    keyword_max_missing_display: int = 10

    # Model settings
    sentence_transformer_model: str = "all-MiniLM-L6-v2"

    # Agent limits
    agent_name_extractor_chars: int = 2000


@lru_cache
def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        pro_model=os.getenv("PRO_MODEL") or "google/gemini-3-pro-preview",
        flash_model=os.getenv("FLASH_MODEL") or "google/gemini-3-flash-preview",
        fast_mode=os.getenv("HR_BREAKER_FAST_MODE", "true").lower() in ("true", "1", "yes"),
        # Scraper settings
        scraper_httpx_timeout=float(os.getenv("SCRAPER_HTTPX_TIMEOUT", "15")),
        scraper_wayback_timeout=float(os.getenv("SCRAPER_WAYBACK_TIMEOUT", "10")),
        scraper_playwright_timeout=int(os.getenv("SCRAPER_PLAYWRIGHT_TIMEOUT", "30000")),
        scraper_httpx_max_retries=int(os.getenv("SCRAPER_HTTPX_MAX_RETRIES", "3")),
        scraper_wayback_max_age_days=int(os.getenv("SCRAPER_WAYBACK_MAX_AGE_DAYS", "30")),
        scraper_min_text_length=int(os.getenv("SCRAPER_MIN_TEXT_LENGTH", "200")),
        # Filter thresholds
        filter_hallucination_threshold=float(os.getenv("FILTER_HALLUCINATION_THRESHOLD", "0.9")),
        filter_keyword_threshold=float(os.getenv("FILTER_KEYWORD_THRESHOLD", "0.25")),
        filter_llm_threshold=float(os.getenv("FILTER_LLM_THRESHOLD", "0.7")),
        filter_vector_threshold=float(os.getenv("FILTER_VECTOR_THRESHOLD", "0.4")),
        filter_ai_generated_threshold=float(os.getenv("FILTER_AI_GENERATED_THRESHOLD", "0.4")),
        # Resume length limits
        resume_max_chars=int(os.getenv("RESUME_MAX_CHARS", "4500")),
        resume_max_words=int(os.getenv("RESUME_MAX_WORDS", "520")),
        resume_page2_overflow_chars=int(os.getenv("RESUME_PAGE2_OVERFLOW_CHARS", "1000")),
        # Keyword matcher params
        keyword_tfidf_max_features=int(os.getenv("KEYWORD_TFIDF_MAX_FEATURES", "200")),
        keyword_tfidf_cutoff=float(os.getenv("KEYWORD_TFIDF_CUTOFF", "0.1")),
        keyword_max_missing_display=int(os.getenv("KEYWORD_MAX_MISSING_DISPLAY", "10")),
        # Model settings
        sentence_transformer_model=os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"),
        # Agent limits
        agent_name_extractor_chars=int(os.getenv("AGENT_NAME_EXTRACTOR_CHARS", "2000")),
    )


def get_model_settings() -> dict[str, Any] | None:
    """Get model settings (currently none for OpenAI-compatible APIs)."""
    return None


def get_pro_model():
    """Get the pro model configured for OpenRouter."""
    from pydantic_ai.models.openai import OpenAIModel

    settings = get_settings()
    return OpenAIModel(
        settings.pro_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def get_flash_model():
    """Get the flash model configured for OpenRouter."""
    from pydantic_ai.models.openai import OpenAIModel

    settings = get_settings()
    return OpenAIModel(
        settings.flash_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
