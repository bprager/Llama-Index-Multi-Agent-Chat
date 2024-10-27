#!/usr/bin/env python3
""" main routine to start the FastAPI server """

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# from app.api.routers import api_router
from app.observability import init_observability
from app.config import ModelSettings


settings = ModelSettings()  # type: ignore [call-arg]


app = FastAPI()

init_observability(app)

ENVIRONMENT = settings.environment or "dev"
logger = logging.getLogger("uvicorn")

if ENVIRONMENT == "dev":
    logger.warning("Running in development mode - allowing CORS for all origins")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Redirect to documentation page when accessing base URL
    @app.get("/")
    async def redirect_to_docs():
        """Redirect to the documentation page"""
        return RedirectResponse(url="/docs")


app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    APP_HOST = settings.app_host
    APP_PORT = settings.app_port
    RELOAD = ENVIRONMENT == "dev"  # True if running in development mode

    uvicorn.run(app="main:app", host=APP_HOST, port=APP_PORT, reload=RELOAD)
