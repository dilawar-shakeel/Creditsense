from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "CreditSense"
    environment: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    database_url: str = "postgresql+psycopg://creditsense:creditsense@localhost:5432/creditsense"
    agent_name: str = "creditsense-agent"
    model_bundle_path: str = "src/creditsense/ml/artifacts/creditsense_bundle.joblib"

    # RAG layer (Phase 4)
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    rerank_model: str = "gpt-4o-mini"
    rag_top_k: int = 5
    rrf_k: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
