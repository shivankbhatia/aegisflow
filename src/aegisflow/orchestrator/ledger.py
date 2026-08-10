import json
from typing import AsyncIterator, Dict, Any, Optional
from psycopg import AsyncConnection, AsyncCursor
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

class PostgresCheckpointer(BaseCheckpointSaver):
    def __init__(self, dsn: str):
        super().__init__()
        self.dsn = dsn
        self.serde = JsonPlusSerializer()

    async def setup(self) -> None:
        async with await AsyncConnection.connect(self.dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute('''
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        thread_id TEXT,
                        checkpoint_id TEXT,
                        checkpoint JSONB,
                        metadata JSONB,
                        PRIMARY KEY (thread_id, checkpoint_id)
                    )
                ''')
            await conn.commit()

    def get_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        # Synchronous get_tuple is not fully implemented in this async mock
        # LangGraph typically expects async checkpointers for async workflows
        raise NotImplementedError("Use aget_tuple instead")

    def put(self, config: dict, checkpoint: Checkpoint, metadata: CheckpointMetadata) -> dict:
        raise NotImplementedError("Use aput instead")

    async def aget_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")

        async with await AsyncConnection.connect(self.dsn) as conn:
            async with conn.cursor() as cur:
                if checkpoint_id:
                    await cur.execute(
                        "SELECT checkpoint, metadata FROM checkpoints WHERE thread_id = %s AND checkpoint_id = %s",
                        (thread_id, checkpoint_id)
                    )
                else:
                    await cur.execute(
                        "SELECT checkpoint, metadata, checkpoint_id FROM checkpoints WHERE thread_id = %s ORDER BY checkpoint_id DESC LIMIT 1",
                        (thread_id,)
                    )
                
                row = await cur.fetchone()
                if not row:
                    return None
                
                cp_data = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                meta_data = json.loads(row[1]) if isinstance(row[1], str) else row[1]
                cp_id = row[2] if len(row) > 2 else checkpoint_id

                return CheckpointTuple(
                    config={"configurable": {"thread_id": thread_id, "checkpoint_id": cp_id}},
                    checkpoint=self.serde.loads(json.dumps(cp_data).encode()),
                    metadata=self.serde.loads(json.dumps(meta_data).encode()),
                    parent_config=None
                )

    async def aput(self, config: dict, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: dict = None) -> dict:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]
        
        async with await AsyncConnection.connect(self.dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO checkpoints (thread_id, checkpoint_id, checkpoint, metadata)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (thread_id, checkpoint_id) DO UPDATE
                    SET checkpoint = EXCLUDED.checkpoint, metadata = EXCLUDED.metadata
                    """,
                    (
                        thread_id,
                        checkpoint_id,
                        self.serde.dumps(checkpoint).decode(),
                        self.serde.dumps(metadata).decode()
                    )
                )
            await conn.commit()
        return {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}

    async def alist(self, config: Optional[dict], *, filter: Optional[Dict[str, Any]] = None, before: Optional[dict] = None, limit: Optional[int] = None) -> AsyncIterator[CheckpointTuple]:
        raise NotImplementedError("alist is not fully implemented for this demo")
