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

    # Agents layer (Phase 5)
    agent_model: str = "gpt-4o-mini"
    ml_api_base_url: str = "http://localhost:8000"
    ml_api_timeout_seconds: float = 10.0
    escalation_probability_band: float = 0.02  # near-cutoff band -> escalate rather than decide
    # 0-10 rerank scale, NOT the RRF-scale OUT_OF_SCOPE_SCORE_FLOOR in rag/eval.py — the two
    # are calibrated for different score scales and must never be swapped. See compliance.py.
    min_rerank_score_for_citation: float = 5.0

    # MCP server (Phase 6). Two fixed bearer tokens rather than a full OAuth
    # authorization server -- see mcp_server/auth.py's docstring for why that's the
    # right amount of auth for this project's scope. Only meaningful on the HTTP
    # transport; stdio has no token concept at all.
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8100
    mcp_read_api_key: str = "creditsense-read-demo-key"
    mcp_write_api_key: str = "creditsense-write-demo-key"
    # The externally-reachable origin the MCP server is actually served from -- NOT
    # mcp_host:mcp_port above, which nothing binds to standalone (api/main.py mounts
    # the MCP app inside the FastAPI app on api_port, at /mcp). OAuth issuer/resource
    # URLs (mcp_server/oauth_provider.py) must exactly match this or discovery fails
    # (RFC 8414/9207 exact-string issuer comparison). Override to a tunnel's https://
    # origin (e.g. ngrok/Cloudflare Tunnel) to connect Claude Desktop's chat connector.
    mcp_public_base_url: str = "http://localhost:8000"

    # Backend integration (Phase 7). Protects the OpenAI bill and the database behind
    # the two write paths (submit_underwriting_decision, /applications/underwrite),
    # not meant as DDoS protection -- see ratelimit.py.
    write_rate_limit_per_minute: int = 10
    write_rate_limit_burst: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
