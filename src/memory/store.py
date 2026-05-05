from typing import List

from src.config.logger import logger
from sqlmodel import Session, select

from src.config.settings import DatabaseSettings
from src.memory.models import Memory, SQLModel
from src.utils.db import get_engine


class MemoryStore:
    """
    Manager for reading/writing Memory records via SQLModel.
    """

    def __init__(self, settings: DatabaseSettings) -> None:
        """
        Initialize the MemoryStore with a database engine and session.

        :param settings: DatabaseSettings with db_url and echo.
        """
        try:
            self.engine = get_engine(settings)
            # Create tables if they do not exist
            SQLModel.metadata.create_all(self.engine)
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            raise

    def store(self, agent_id: str, step: str, content: str) -> Memory:
        """
        Store a new memory record.

        :param agent_id: Identifier of the agent
        :param step: Action or step name
        :param content: Text content to store
        :return: The created Memory instance
        """
        memory = Memory(agent_id=agent_id, step=step, content=content)
        try:
            with Session(self.engine) as session:
                session.add(memory)
                session.commit()
                session.refresh(memory)
            logger.debug(f"Stored memory: {memory}")
            return memory
        except Exception as e:
            logger.error(f"Failed to store memory for {agent_id}:{step}: {e}")
            raise

    def load(self, agent_id: str) -> List[Memory]:
        """
        Load all memory records for a given agent.

        :param agent_id: Identifier of the agent
        :return: List of Memory instances
        """
        try:
            with Session(self.engine) as session:
                statement = select(Memory).where(Memory.agent_id == agent_id)
                results = session.exec(statement).all()
            logger.debug(f"Loaded {len(results)} memories for agent_id={agent_id}")
            return results
        except Exception as e:
            logger.error(f"Failed to load memories for {agent_id}: {e}")
            raise

    def touch(self, memory_id: int) -> None:
        """
        Refresh a memory row's timestamp to ``now``.

        Used by the scrape probe-and-compare flow: when Overpass returns
        identical data after the cache TTL expires, we extend the existing
        row's lifetime instead of inserting a duplicate.

        :param memory_id: Primary key of the row to touch.
        """
        from datetime import datetime, timezone

        try:
            with Session(self.engine) as session:
                row = session.get(Memory, memory_id)
                if row is None:
                    logger.warning(f"touch: memory id={memory_id} not found")
                    return
                row.timestamp = datetime.now(timezone.utc)
                session.add(row)
                session.commit()
            logger.debug(f"Touched memory id={memory_id}")
        except Exception as e:
            logger.error(f"Failed to touch memory id={memory_id}: {e}")
            raise
