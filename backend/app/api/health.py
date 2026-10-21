from fastapi import APIRouter
from app.api.deps import SettingsDep

router = APIRouter(tags=["health"])

@router.get("/health")
def health(settings: SettingsDep):
    return {"ok": True, "app": settings.app_name, "environment": settings.environment}