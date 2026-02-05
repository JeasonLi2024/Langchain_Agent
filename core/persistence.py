
import pickle
import redis
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)

class PickleRedisSaver(BaseCheckpointSaver):
    """
    A custom Redis CheckpointSaver that uses Pickle serialization.
    This avoids the requirement for RedisJSON module.
    It also implements async methods by wrapping sync calls, ensuring compatibility with async graphs.
    """
    def __init__(self, client: redis.Redis):
        super().__init__()
        self.client = client

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"].get("thread_id")
        checkpoint_id = config["configurable"].get("checkpoint_id")
        
        if not thread_id:
            return None
            
        if checkpoint_id:
            key = f"checkpoint:{thread_id}:{checkpoint_id}"
            data = self.client.get(key)
            if data:
                return pickle.loads(data)
        else:
            # Get latest
            latest_key = f"checkpoint_latest:{thread_id}"
            latest_id = self.client.get(latest_key)
            if latest_id:
                key = f"checkpoint:{thread_id}:{latest_id.decode()}"
                data = self.client.get(key)
                if data:
                    return pickle.loads(data)
        return None

    def list(self, config: Optional[RunnableConfig], *, filter: Optional[Dict[str, Any]] = None, before: Optional[RunnableConfig] = None, limit: Optional[int] = None) -> Iterator[CheckpointTuple]:
        # Simplified: iterate is not fully implemented
        return iter([])

    def put(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: dict) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]
        
        parent_id = config["configurable"].get("checkpoint_id")
        
        # Construct the tuple to save
        tuple_data = CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config={"configurable": {"thread_id": thread_id, "checkpoint_id": parent_id}} if parent_id else None,
            pending_writes=[] # We don't store pending writes in this simple version
        )
        
        key = f"checkpoint:{thread_id}:{checkpoint_id}"
        self.client.set(key, pickle.dumps(tuple_data))
        
        # Update latest pointer
        self.client.set(f"checkpoint_latest:{thread_id}", checkpoint_id)
        
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(self, config: RunnableConfig, writes: Sequence[Tuple[str, Any]], task_id: str) -> None:
        # Simplified: We don't persist intermediate writes for now.
        # This is acceptable for "Conversation History" persistence.
        pass
        
    # --- Async Implementations (Sync-over-Async) ---
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return self.get_tuple(config)

    async def aput(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: dict) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)
        
    async def aput_writes(self, config: RunnableConfig, writes: Sequence[Tuple[str, Any]], task_id: str) -> None:
        return self.put_writes(config, writes, task_id)
