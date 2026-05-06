"""
Pipeline execution endpoints.

This module provides endpoints for running the complete surveillance analysis pipeline,
including scraping, analysis, and optional routing.
"""

import asyncio
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.api.models.requests import PipelineRequest
from src.api.models.responses import TaskResponse, TaskStatus
from src.api.services.task_manager import task_manager
from src.api.services.websocket_manager import ws_manager
from src.orchestration.langchain_pipeline import create_pipeline
from src.config.logger import logger

router = APIRouter(prefix="/pipeline")


async def _broadcast_cache_status(task_id: str, scrape_result) -> None:
    """
    Broadcast a `cache_status` WebSocket event derived from the scraper result.

    Carries probe-and-compare metadata so the frontend can distinguish:
      - Fresh cache hit (no Overpass call): "Showing cached data, 4h old"
      - Probed, unchanged: "OSM returned the same N cameras; using cached
        analysis"
      - Probed, changed: "OSM returned N cameras (Δ +6); refreshing analysis"

    Skipped if the scraper did not run or did not produce cache metadata
    (e.g. build-stage failure before any cache decision was made).
    """
    if not isinstance(scrape_result, dict) or "cache_hit" not in scrape_result:
        return

    cache_hit = scrape_result["cache_hit"]
    probed = scrape_result.get("probed", False)
    changed = scrape_result.get("changed", not cache_hit)
    elements = scrape_result.get("elements_count", 0)
    previous = scrape_result.get("previous_elements_count")
    delta = scrape_result.get("delta")
    age = scrape_result.get("data_age_hours")

    if probed and not changed:
        message = (
            f"OpenStreetMap returned the same {elements} cameras — "
            f"using cached analysis."
        )
    elif probed and changed:
        if delta is not None:
            sign = "+" if delta > 0 else ""
            message = (
                f"OpenStreetMap returned {elements} cameras "
                f"({sign}{delta} vs previous {previous}); refreshing analysis."
            )
        else:
            message = f"Fetched {elements} cameras from OpenStreetMap (first scan)."
    elif cache_hit:
        age_txt = f"{age:.1f}h old" if age is not None else "cached"
        message = f"Loaded {elements} cameras from cache ({age_txt})."
    else:
        message = f"Fetched {elements} cameras from OpenStreetMap."

    await ws_manager.broadcast_progress(
        task_id,
        {
            "type": "cache_status",
            "stage": "scraping",
            "cache_hit": cache_hit,
            "probed": probed,
            "changed": changed,
            "elements_count": elements,
            "previous_elements_count": previous,
            "delta": delta,
            "cached_at": scrape_result.get("cached_at"),
            "data_age_hours": age,
            "cache_ttl_hours": scrape_result.get("cache_ttl_hours"),
            "cache_expires_at": scrape_result.get("cache_expires_at"),
            "message": message,
            "timestamp": datetime.now().isoformat(),
        },
    )


async def _broadcast_analysis_starting(
    task_id: str,
    elements_count: int | None,
    will_skip: bool,
) -> None:
    """
    Emit ``analysis_starting`` once the scrape stage returns and we know
    the element count + planned analyzer-skip decision.

    Advisory only — the polling-based UI consumes the same data via
    ``task.metadata`` (`elements_count`, `analysis_skipped`). Kept as a
    WebSocket event so any future WS consumer can read the count without
    parsing strings.
    """
    if will_skip:
        message = (
            f"Reusing prior analysis of {elements_count} cameras."
            if elements_count is not None
            else "Reusing prior analysis."
        )
    else:
        message = (
            f"Analyzing {elements_count} cameras…"
            if elements_count is not None
            else "Analyzing surveillance infrastructure"
        )

    await ws_manager.broadcast_progress(
        task_id,
        {
            "type": "analysis_starting",
            "stage": "analyzing",
            "progress": 50,
            "elements_count": elements_count,
            "skipped": will_skip,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        },
    )


async def _broadcast_analysis_skipped(task_id: str, analyze_result) -> None:
    """
    Emit `analysis_skipped` when the orchestrator reused prior enriched
    outputs because the scrape probe reported no change. Lets the UI swap
    the "Analyzing…" spinner for a "Reusing prior analysis" state.
    """
    if not isinstance(analyze_result, dict):
        return
    if analyze_result.get("skipped_reason") != "scrape_unchanged":
        return

    await ws_manager.broadcast_progress(
        task_id,
        {
            "type": "analysis_skipped",
            "stage": "analyzing",
            "reason": "scrape_unchanged",
            "element_count": analyze_result.get("element_count", 0),
            "geojson_path": analyze_result.get("geojson_path"),
            "message": (
                "No new cameras since the last scan; reusing prior analysis outputs."
            ),
            "timestamp": datetime.now().isoformat(),
        },
    )


async def _check_and_broadcast_cancellation(task_id: str, stage: str) -> bool:
    """
    Check if task is cancelled and broadcast cancellation message.

    :param task_id: Task identifier
    :param stage: Current pipeline stage for logging
    :return: True if task was cancelled, False otherwise
    """
    if task_manager.is_cancelled(task_id):
        logger.info(f"Pipeline task {task_id} cancelled at stage: {stage}")
        await ws_manager.broadcast_progress(
            task_id,
            {
                "type": "cancelled",
                "stage": "cancelled",
                "progress": 0,
                "message": "Pipeline cancelled by user",
                "timestamp": datetime.now().isoformat(),
            },
        )
        return True
    return False


async def execute_pipeline_task(task_id: str, request: PipelineRequest) -> None:
    """
    Execute pipeline in background with real-time progress updates.

    This function runs the SurveillancePipeline asynchronously and broadcasts
    progress updates via WebSocket to connected clients.

    :param task_id: Task identifier
    :param request: Pipeline request parameters
    """
    try:
        task_manager.mark_running(task_id)
        logger.info(f"Starting pipeline task {task_id} for {request.city}")

        # Broadcast initialization
        await ws_manager.broadcast_progress(
            task_id,
            {
                "type": "progress",
                "stage": "initializing",
                "progress": 0,
                "message": f"Initializing pipeline for {request.city}",
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Check for cancellation before starting
        if await _check_and_broadcast_cancellation(task_id, "initialization"):
            return

        # Build configuration from request
        config_kwargs = {"force_refresh": request.force_refresh}

        # Add routing config if provided
        if request.routing_config:
            config_kwargs.update(
                {
                    "routing_enabled": True,
                    "start_lat": request.routing_config.start_lat,
                    "start_lon": request.routing_config.start_lon,
                    "end_lat": request.routing_config.end_lat,
                    "end_lon": request.routing_config.end_lon,
                }
            )

        # Check for cancellation before scraping
        if await _check_and_broadcast_cancellation(task_id, "scraping"):
            return

        # Broadcast scraping stage
        task_manager.update_progress(task_id, 20, "Scraping surveillance data...")
        await ws_manager.broadcast_progress(
            task_id,
            {
                "type": "progress",
                "stage": "scraping",
                "progress": 20,
                "message": "Downloading surveillance data from OpenStreetMap",
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Create and run pipeline with cancellation support
        pipeline = create_pipeline(request.scenario, **config_kwargs)
        pipeline.cancellation_check = lambda: task_manager.is_cancelled(task_id)

        # Wire the on-scrape-complete hook. The pipeline runs on a worker
        # thread (via asyncio.to_thread) so we capture the running loop
        # here and use run_coroutine_threadsafe to schedule the WS
        # broadcast back on it. ``task_manager`` mutations are sync dict
        # ops and safe to call from any thread.
        loop = asyncio.get_running_loop()

        def _on_scrape_complete(payload: dict) -> None:
            elements_count = payload.get("elements_count")
            will_skip = bool(payload.get("will_skip_analyzer"))

            if will_skip:
                msg = (
                    f"Reusing prior analysis of {elements_count} cameras."
                    if elements_count is not None
                    else "Reusing prior analysis."
                )
            else:
                msg = (
                    f"Analyzing {elements_count} cameras…"
                    if elements_count is not None
                    else "Analyzing data..."
                )

            task_manager.update_progress(task_id, 50, msg)
            metadata_fields: dict = {"analysis_skipped": will_skip}
            if elements_count is not None:
                metadata_fields["elements_count"] = elements_count
            task_manager.set_metadata(task_id, **metadata_fields)

            asyncio.run_coroutine_threadsafe(
                _broadcast_analysis_starting(task_id, elements_count, will_skip),
                loop,
            )

        pipeline.on_scrape_complete = _on_scrape_complete

        run_kwargs = {}
        if request.country:
            run_kwargs["country"] = request.country

        # Check for cancellation before analysis
        if await _check_and_broadcast_cancellation(task_id, "analysis"):
            return

        # Execute the pipeline in a thread pool to avoid blocking the event loop
        # This allows cancellation checks to respond immediately. The
        # analyzer-stage progress broadcast happens from the
        # on_scrape_complete callback once we know the element count.
        results = await asyncio.to_thread(pipeline.run, request.city, **run_kwargs)

        # Surface scrape cache state and analyzer skip-on-unchanged to the
        # frontend so it can render a "showing cached data" badge, a delta
        # readout (+6 cameras since last scan), and a refresh affordance.
        # Both events are no-ops when the corresponding stage didn't run or
        # didn't produce metadata.
        await _broadcast_cache_status(task_id, results.get("scrape"))
        await _broadcast_analysis_skipped(task_id, results.get("analyze"))

        # Check the actual pipeline status and handle accordingly
        status = results.get("status")

        if status == "cancelled":
            # Task was cancelled during execution
            task_manager.mark_cancelled(task_id)
            await ws_manager.broadcast_progress(
                task_id,
                {
                    "type": "cancelled",
                    "stage": "cancelled",
                    "progress": 0,
                    "message": "Pipeline cancelled by user",
                    "timestamp": datetime.now().isoformat(),
                },
            )
            logger.info(f"Pipeline task {task_id} cancelled")
        elif status in ["failed", "partial"]:
            # Task failed or partially completed
            error_msg = results.get("error", "Unknown error")
            task_manager.mark_failed(task_id, error_msg)
            await ws_manager.broadcast_progress(
                task_id,
                {
                    "type": "failed",
                    "stage": "failed",
                    "progress": 0,
                    "message": f"Pipeline {status}: {error_msg}",
                    "timestamp": datetime.now().isoformat(),
                },
            )
            logger.warning(f"Pipeline task {task_id} {status}: {error_msg}")
        else:
            # Task completed successfully
            task_manager.mark_completed(task_id, results)
            await ws_manager.broadcast_progress(
                task_id,
                {
                    "type": "completed",
                    "stage": "completed",
                    "progress": 100,
                    "message": "Pipeline completed successfully",
                    "timestamp": datetime.now().isoformat(),
                },
            )
            logger.info(f"Pipeline task {task_id} completed successfully")

    except Exception as e:
        logger.error(f"Pipeline task {task_id} failed: {e}")
        task_manager.mark_failed(task_id, str(e))

        # Broadcast failure
        await ws_manager.broadcast_progress(
            task_id,
            {
                "type": "failed",
                "stage": "failed",
                "progress": 0,
                "message": f"Pipeline failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
            },
        )


@router.post("/run", response_model=TaskResponse)
async def run_pipeline(
    request: PipelineRequest, background_tasks: BackgroundTasks
) -> TaskResponse:
    """
    Start a complete pipeline execution.

    The pipeline will run in the background. Use the returned task_id
    to check status via GET /api/v1/pipeline/{task_id}.

    :param request: Pipeline configuration
    :param background_tasks: FastAPI background tasks handler
    :return: Task creation response with task_id
    """
    # Create task
    task_id = task_manager.create_task(
        "pipeline",
        metadata={
            "city": request.city,
            "country": request.country,
            "scenario": request.scenario.value,
        },
    )

    # Schedule background execution
    background_tasks.add_task(execute_pipeline_task, task_id, request)

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message=f"Pipeline started for {request.city}",
    )


@router.get("/{task_id}")
async def get_pipeline_status(task_id: str):
    """
    Get pipeline task status and results.

    :param task_id: Task identifier
    :return: Task status with results (if completed)
    :raises HTTPException: 404 if task not found
    """
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return task.to_dict()


@router.post("/{task_id}/cancel")
async def cancel_pipeline(task_id: str):
    """
    Cancel a running pipeline task.

    This endpoint uses cooperative cancellation: the task is marked as cancelled,
    and the background task checks for cancellation at key stages (initialization,
    scraping, analysis). The task will exit gracefully at the next checkpoint.

    :param task_id: Task identifier
    :return: Cancellation confirmation
    :raises HTTPException: 404 if task not found, 400 if task already completed/failed
    """
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
        raise HTTPException(
            status_code=400, detail=f"Cannot cancel task in {task.status.value} state"
        )

    task_manager.mark_cancelled(task_id)

    return {
        "task_id": task_id,
        "status": "cancelled",
        "message": "Task cancelled successfully",
    }


@router.delete("/{task_id}")
async def delete_pipeline_task(task_id: str):
    """
    Delete a pipeline task and its results.

    :param task_id: Task identifier
    :return: Deletion confirmation
    :raises HTTPException: 404 if task not found
    """
    if not task_manager.delete_task(task_id):
        raise HTTPException(status_code=404, detail="Task not found")

    return {"task_id": task_id, "message": "Task deleted successfully"}
