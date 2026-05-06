"""
Tests for TaskManager service.

These tests verify the in-memory task management functionality.
"""

import pytest

from src.api.services.task_manager import TaskManager
from src.api.models.responses import TaskStatus


@pytest.fixture
def task_manager():
    """Create a fresh TaskManager instance for each test."""
    return TaskManager()


def test_create_task(task_manager):
    """Test creating a new task."""
    task_id = task_manager.create_task("pipeline", metadata={"city": "Berlin"})

    assert task_id is not None
    assert len(task_id) > 0

    task = task_manager.get_task(task_id)
    assert task is not None
    assert task.type == "pipeline"
    assert task.status == TaskStatus.PENDING
    assert task.metadata["city"] == "Berlin"


def test_get_nonexistent_task(task_manager):
    """Test getting a task that doesn't exist."""
    task = task_manager.get_task("nonexistent-id")
    assert task is None


def test_mark_task_running(task_manager):
    """Test marking a task as running."""
    task_id = task_manager.create_task("pipeline")
    task_manager.mark_running(task_id)

    task = task_manager.get_task(task_id)
    assert task.status == TaskStatus.RUNNING
    assert task.started_at is not None


def test_mark_task_completed(task_manager):
    """Test marking a task as completed with result."""
    task_id = task_manager.create_task("pipeline")
    result_data = {"status": "success", "files": ["output.json"]}

    task_manager.mark_completed(task_id, result_data)

    task = task_manager.get_task(task_id)
    assert task.status == TaskStatus.COMPLETED
    assert task.progress == 100
    assert task.result == result_data
    assert task.completed_at is not None


def test_mark_task_failed(task_manager):
    """Test marking a task as failed with error."""
    task_id = task_manager.create_task("pipeline")
    error_message = "Connection timeout"

    task_manager.mark_failed(task_id, error_message)

    task = task_manager.get_task(task_id)
    assert task.status == TaskStatus.FAILED
    assert task.error == error_message
    assert task.completed_at is not None


def test_mark_task_cancelled(task_manager):
    """Test marking a task as cancelled."""
    task_id = task_manager.create_task("pipeline")
    task_manager.mark_cancelled(task_id)

    task = task_manager.get_task(task_id)
    assert task.status == TaskStatus.CANCELLED
    assert task.completed_at is not None


def test_is_cancelled(task_manager):
    """Test checking if a task is cancelled."""
    # Create and run a task
    task_id = task_manager.create_task("pipeline")
    task_manager.mark_running(task_id)

    # Should not be cancelled initially
    assert task_manager.is_cancelled(task_id) is False

    # Cancel the task
    task_manager.mark_cancelled(task_id)

    # Should be cancelled now
    assert task_manager.is_cancelled(task_id) is True

    # Non-existent task should return False
    assert task_manager.is_cancelled("nonexistent-id") is False


def test_update_task_progress(task_manager):
    """Test updating task progress."""
    task_id = task_manager.create_task("pipeline")
    task_manager.update_progress(task_id, 50, "Processing data...")

    task = task_manager.get_task(task_id)
    assert task.progress == 50
    assert task.metadata["last_message"] == "Processing data..."


def test_set_metadata_merges_typed_fields(task_manager):
    """``set_metadata`` should merge fields without dropping existing keys."""
    task_id = task_manager.create_task("pipeline", metadata={"city": "Lund"})
    task_manager.update_progress(task_id, 50, "Analyzing 152 cameras…")

    task_manager.set_metadata(task_id, elements_count=152, analysis_skipped=False)

    task = task_manager.get_task(task_id)
    assert task.metadata["city"] == "Lund"  # preserved
    assert task.metadata["last_message"] == "Analyzing 152 cameras…"  # preserved
    assert task.metadata["elements_count"] == 152
    assert task.metadata["analysis_skipped"] is False


def test_set_metadata_overwrites_existing_keys(task_manager):
    """Subsequent ``set_metadata`` calls overwrite the same keys."""
    task_id = task_manager.create_task("pipeline")
    task_manager.set_metadata(task_id, elements_count=10, analysis_skipped=True)
    task_manager.set_metadata(task_id, elements_count=99)

    task = task_manager.get_task(task_id)
    assert task.metadata["elements_count"] == 99
    assert task.metadata["analysis_skipped"] is True  # untouched


def test_set_metadata_on_nonexistent_task_is_noop(task_manager):
    """Setting metadata on a missing task should not raise."""
    task_manager.set_metadata("nonexistent-id", elements_count=42)  # no exception


def test_delete_task(task_manager):
    """Test deleting a task."""
    task_id = task_manager.create_task("pipeline")

    # Verify task exists
    assert task_manager.get_task(task_id) is not None

    # Delete task
    result = task_manager.delete_task(task_id)
    assert result is True

    # Verify task is gone
    assert task_manager.get_task(task_id) is None


def test_delete_nonexistent_task(task_manager):
    """Test deleting a task that doesn't exist."""
    result = task_manager.delete_task("nonexistent-id")
    assert result is False


def test_task_to_dict(task_manager):
    """Test converting task to dictionary."""
    task_id = task_manager.create_task(
        "pipeline", metadata={"city": "Berlin", "scenario": "basic"}
    )
    task = task_manager.get_task(task_id)
    task_dict = task.to_dict()

    assert task_dict["id"] == task_id
    assert task_dict["type"] == "pipeline"
    assert task_dict["status"] == "pending"
    assert task_dict["progress"] == 0
    assert "created_at" in task_dict
    assert task_dict["metadata"]["city"] == "Berlin"


def test_multiple_tasks(task_manager):
    """Test managing multiple tasks simultaneously."""
    task_id_1 = task_manager.create_task("scrape", metadata={"city": "Berlin"})
    task_id_2 = task_manager.create_task("analyze", metadata={"city": "Athens"})

    task_manager.mark_running(task_id_1)
    task_manager.mark_completed(task_id_2, {"status": "done"})

    task_1 = task_manager.get_task(task_id_1)
    task_2 = task_manager.get_task(task_id_2)

    assert task_1.status == TaskStatus.RUNNING
    assert task_2.status == TaskStatus.COMPLETED
    assert task_1.metadata["city"] == "Berlin"
    assert task_2.metadata["city"] == "Athens"
