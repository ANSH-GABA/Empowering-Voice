from fastapi import APIRouter
from app.api.deps import SettingsDep

router = APIRouter()          # no prefix here

@router.get("/ping")
def ping(settings: SettingsDep):
    return {"recognizer": settings.recognizer}