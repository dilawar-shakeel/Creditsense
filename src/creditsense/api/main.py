from fastapi import FastAPI

from creditsense.api.predict import router as prediction_router
from creditsense.config import get_settings

settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0")
app.include_router(prediction_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name, "environment": settings.environment}
