from fastapi import APIRouter
from app.api.v1 import recognize   
# add attempts, learners, catalog, lessons as you write them

api_v1_router = APIRouter()
api_v1_router.include_router(recognize.router, prefix="/recognize", tags=["recognize"])