"""
FastAPI application entry point.

This module creates and configures the FastAPI application, including
middleware, routers, and static file serving.

"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes import health
from src.api.routes import pipeline
from src.api.routes import outputs
from src.api.services.websocket_manager import ws_manager
from src.api.services.task_manager import task_manager
from src.config.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for startup and shutdown tasks.

    :param app: FastAPI application instance
    """
    # Startup
    logger.info("FastAPI server starting up")
    logger.info("API documentation available at /docs and /redoc")

    yield

    # Shutdown
    logger.info("FastAPI server shutting down")


# Create FastAPI app
app = FastAPI(
    title="Agentic Surveillance Research API",
    description="""
    REST API for analyzing surveillance infrastructure and computing
    privacy-preserving walking routes.

    ## Features
    - **Asynchronous job execution** for long-running operations
    - **Real-time progress updates** via WebSocket at `/ws/tasks/{task_id}`
    - **GeoJSON and interactive map outputs**
    - **Intelligent caching** to avoid redundant computation
    - **Multiple analysis scenarios** (basic, full, quick, report, mapping)

    ## Architecture
    This API layer imports and uses the existing `SurveillancePipeline` class
    that powers the CLI.

    ## Usage
    1. Start a pipeline job via POST /api/v1/pipeline/run
    2. Monitor progress via GET /api/v1/pipeline/{task_id}
    3. Retrieve generated files via /api/v1/outputs/...
    """,
    version="2.4.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount static files for outputs
# This will be available at /outputs/* and serves files from overpass_data/
app.mount("/outputs", StaticFiles(directory="overpass_data"), name="outputs")


# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(pipeline.router, prefix="/api/v1", tags=["pipeline"])
app.include_router(outputs.router, prefix="/api/v1", tags=["outputs"])


# WebSocket endpoint for real-time progress updates
@app.websocket("/ws/tasks/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for real-time task progress updates.

    Connect to this endpoint to receive live updates about task execution.
    Messages are JSON formatted with structure:
    {
        "type": "status" | "progress" | "completed" | "failed",
        "status": "pending" | "running" | "completed" | "failed",
        "progress": 0-100,
        "message": "optional message",
        "timestamp": "ISO 8601 timestamp"
    }

    :param websocket: WebSocket connection
    :param task_id: Task identifier to monitor
    """
    await ws_manager.connect(task_id, websocket)
    try:
        # Keep connection alive and echo task status on client requests
        while True:
            # Wait for client message (keep-alive or status request)
            _ = await websocket.receive_text()

            # Send current task status
            task = task_manager.get_task(task_id)
            if task:
                await websocket.send_json(
                    {
                        "type": "status",
                        "status": task.status.value,
                        "progress": task.progress,
                        "task_id": task_id,
                    }
                )
            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Task not found",
                        "task_id": task_id,
                    }
                )
                break

    except WebSocketDisconnect:
        ws_manager.disconnect(task_id, websocket)
        logger.debug(f"WebSocket disconnected for task {task_id}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )
