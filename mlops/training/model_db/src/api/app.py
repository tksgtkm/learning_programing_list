from logging import getLogger

from fastapi import FastAPI
from src.api.routers import api, health
from src.configurations import APIconfigrations
from src.db import initialize
from src.db.database import engine

logger = getLogger(__name__)

initialize.initialize_table(engine=engine, checkfirst=True)

app = FastAPI(
    title=APIconfigrations.title,
    description=APIconfigrations.description,
    version=APIconfigrations.version
)

app.include_router(health.router, prefix=f"/v{APIconfigrations.version}/health", tags=["health"])
app.include_router(api.router, prefix=f"/v{APIconfigrations.version}/api", tags=["api"])