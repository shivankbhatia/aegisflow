import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Callable

from dotenv import load_dotenv

from ..llm_gateway import from_env
from ..orchestrator.guardrails import GuardrailsInterceptor
from ..orchestrator.workflow import Agents, build_workflow, default_agents

AgentsFactory = Callable[[GuardrailsInterceptor], Agents]


@dataclass
class RunRecord:
    run_id: str
    events: list[dict] = field(default_factory=list)
    finished: bool = False
    error: str | None = None
    task: asyncio.Task | None = None
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()


class RunManager:
    """Runs workflows in the background; durable state lives in Postgres,
    live events and error text live in memory (lost on restart)."""

    def __init__(self, checkpointer, agents_factory: AgentsFactory | None = None):
        self.checkpointer = checkpointer
        self.agents_factory = agents_factory
        self.records: dict[str, RunRecord] = {}
        self._client = None  # lazily created, shared by all runs
        # Read-only graph for state lookups; its nodes are never executed.
        self._reader = build_workflow(
            agents=Agents(None, None, None), checkpointer=checkpointer
        )

    # ---- internals -------------------------------------------------------

    def _default_factory(self, guardrails: GuardrailsInterceptor) -> Agents:
        if self._client is None:
            load_dotenv()
            self._client = from_env()
        return default_agents(guardrails, self._client)

    def _graph(self):
        guardrails = GuardrailsInterceptor()
        factory = self.agents_factory or self._default_factory
        return build_workflow(factory(guardrails), self.checkpointer, guardrails)

    async def _emit(self, rec: RunRecord, event: dict):
        async with rec.cond:
            rec.events.append({"seq": len(rec.events), **event})
            rec.cond.notify_all()

    async def _run(self, rec: RunRecord, graph_input):
        config = {"configurable": {"thread_id": rec.run_id}}
        try:
            graph = self._graph()
            async for update in graph.astream(graph_input, config, stream_mode="updates"):
                for node, delta in update.items():
                    await self._emit(rec, {"event": "node", "node": node, "update": delta})
            await self._emit(rec, {"event": "done"})
        except Exception as e:
            rec.error = f"{type(e).__name__}: {e}"
            await self._emit(rec, {"event": "error", "detail": rec.error})
        finally:
            async with rec.cond:
                rec.finished = True
                rec.cond.notify_all()

    def _launch(self, rec: RunRecord, graph_input):
        rec.finished, rec.error = False, None
        rec.task = asyncio.create_task(self._run(rec, graph_input))

    # ---- public API ------------------------------------------------------

    def start(self, task: str) -> str:
        run_id = str(uuid.uuid4())
        rec = self.records[run_id] = RunRecord(run_id)
        self._launch(rec, {"task": task, "recursion_count": 0, "tokens_used": 0})
        return run_id

    def resume(self, run_id: str):
        rec = self.records.get(run_id) or self.records.setdefault(run_id, RunRecord(run_id))
        self._launch(rec, None)  # input=None: continue from last checkpoint

    async def status(self, run_id: str) -> dict | None:
        rec = self.records.get(run_id)
        snap = await self._reader.aget_state({"configurable": {"thread_id": run_id}})
        v = snap.values or {}
        if not v and rec is None:
            return None
        if rec and rec.running:
            status = "running"
        elif snap.next:
            status = "failed" if (rec and rec.error) else "interrupted"
        elif v:
            status = "completed"
        elif rec and rec.error:
            status = "failed"
        else:
            status = "queued"
        return {
            "run_id": run_id,
            "status": status,
            "error": rec.error if rec else None,
            "next": list(snap.next),
            "approved": v.get("approved"),
            "feedback": v.get("feedback"),
            "plan": v.get("plan"),
            "code_patch": v.get("code_patch"),
            "test_patch": v.get("test_patch"),
            "attempts": v.get("recursion_count", 0),
            "tokens_used": v.get("tokens_used", 0),
        }

    async def aclose(self):
        for rec in self.records.values():
            if rec.running:
                rec.task.cancel()
        if self._client is not None:
            await self._client.aclose()