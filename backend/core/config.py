from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",   # absolute path: works from any folder
        env_prefix="EV_",                # EV_DATABASE_URL, EV_LLM_API_KEY, ...
        extra="ignore",
    )

    # app
    app_name: str = "Empowering Voice API"
    environment: Literal["dev", "test", "prod"] = "dev"
    api_v1_prefix: str = "/api/v1"
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:5173"]

    # data
    database_url: str = "sqlite:///./empowering_voice.db"

    # recognition
    recognizer: Literal["hf", "landmark"] = "landmark"
    hf_model: str = "Hemgg/sign-language-classification"
    skip_warmup: bool = False
    max_image_bytes: int = Field(2_000_000, gt=0)

    # learning thresholds
    min_confidence: float = Field(0.8, ge=0, le=1)
    mastery_weight: float = Field(0.3, gt=0, le=1)

    # agents / LLM (secret never leaves the server)
    llm_api_key: SecretStr | None = None
    llm_model_name: str = "set-me-in-env"
    llm_timeout_s: float = Field(8.0, gt=0)
    llm_daily_limit: int = Field(50, ge=0)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @model_validator(mode="after")
    def prod_guards(self):
        if self.environment == "prod" and "*" in self.cors_origins:
            raise ValueError("Wildcard CORS origin is not allowed in prod")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()