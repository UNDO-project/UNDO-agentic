"""
Task management service for background job tracking.

This module provides in-memory task tracking for asynchronous pipeline operations.
Tasks can be created, updated, and queried by their unique identifiers.
"""

from typing import Dict, Any, Optional
from uuid import uuid4
from datetime import datetime

from src.api.models.responses import TaskStatus


class Task:
    """
    Represents a background task with status tracking.

    :param task_id: Unique identifier for the task
    :param task_type: Type of task (scrape, analyze, route, pipeline)
    """

    def __init__(self, task_id: str, task_type: str):
        """
        Initialize a new task.

        :param task_id: Unique task identifier
        :param task_type: Type of task being executed
        """
        self.id = task_id
        self.type = task_type
        self.status = TaskStatus.PENDING
        self.progress = 0
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert task to dictionary representation.

        :return: Dictionary with all task fields
        """
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status.value,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "metadata": self.metadata,
        }


class TaskManager:
    """
    Manages background tasks with in-memory storage.

    This is a simple in-memory implementation. For production with multiple
    workers, consider using Redis or a database for task storage.
    """

    def __init__(self):
        """Initialize the task manager with empty task storage."""
        self.tasks: Dict[str, Task] = {}

    def create_task(
        self, task_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new task.

        :param task_type: Type of task (scrape, analyze, route, pipeline)
        :param metadata: Optional metadata for the task
        :return: Task ID (UUID string)
        """
        task_id = str(uuid4())
        task = Task(task_id, task_type)
        if metadata:
            task.metadata = metadata
        self.tasks[task_id] = task
        return task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Retrieve a task by ID.

        :param task_id: Task identifier
        :return: Task object if found, None otherwise
        """
        return self.tasks.get(task_id)

    def update_progress(
        self, task_id: str, progress: int, message: Optional[str] = None
    ) -> None:
        """
        Update task progress.

        :param task_id: Task identifier
        :param progress: Progress percentage (0-100)
        :param message: Optional progress message
        """
        if task := self.tasks.get(task_id):
            task.progress = progress
            if message:
                task.metadata["last_message"] = message

    def set_metadata(self, task_id: str, **fields: Any) -> None:
        """
        Merge typed metadata fields into ``task.metadata``.

        Used to surface structured signals on the polled task response
        (e.g. ``elements_count``, ``analysis_skipped``) without abusing
        ``last_message`` as a string transport. Existing keys are
        overwritten; other keys are left intact.

        :param task_id: Task identifier
        :param fields: Arbitrary keyword arguments merged into metadata
        """
        if task := self.tasks.get(task_id):
            task.metadata.update(fields)

    def mark_running(self, task_id: str) -> None:
        """
        Mark task as running.

        :param task_id: Task identifier
        """
        if task := self.tasks.get(task_id):
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()

    def mark_completed(self, task_id: str, result: Any) -> None:
        """
        Mark task as completed with result.

        :param task_id: Task identifier
        :param result: Task result data
        """
        if task := self.tasks.get(task_id):
            task.status = TaskStatus.COMPLETED
            task.progress = 100
            task.result = result
            task.completed_at = datetime.now()

    def mark_failed(self, task_id: str, error: str) -> None:
        """
        Mark task as failed with error message.

        :param task_id: Task identifier
        :param error: Error message
        """
        if task := self.tasks.get(task_id):
            task.status = TaskStatus.FAILED
            task.error = error
            task.completed_at = datetime.now()

    def mark_cancelled(self, task_id: str) -> None:
        """
        Mark task as cancelled.

        :param task_id: Task identifier
        """
        if task := self.tasks.get(task_id):
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()

    def is_cancelled(self, task_id: str) -> bool:
        """
        Check if task has been cancelled.

        :param task_id: Task identifier
        :return: True if task is cancelled, False otherwise
        """
        task = self.tasks.get(task_id)
        return task.status == TaskStatus.CANCELLED if task else False

    def delete_task(self, task_id: str) -> bool:
        """
        Delete a task from storage.

        :param task_id: Task identifier
        :return: True if task was deleted, False if not found
        """
        if task_id in self.tasks:
            del self.tasks[task_id]
            return True
        return False


# Global task manager instance
task_manager = TaskManager()
