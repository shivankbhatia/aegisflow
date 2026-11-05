import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Callable

from dotenv import load_dotenv
from langgraph.types import Command

from ..llm_gateway import from_env
from ..orchestrator.guardrails import GuardrailsInterceptor
from ..orchestrator.workflow import Agents, build_workflow, default_agents
from ..rag.qa_workflow import build_qa_workflow

AgentsFactory = Callable[[GuardrailsInterceptor], Agents]


@dataclass
class RunRecord:
    run_id: str
    graph_name: str = "code"
    events: list[dict] = field(default_factory=list)
    finished: bool = False
    error: str | None = None
    awaiting_approval: bool = False
    task: asyncio.Task | None = None
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()


class RunManager:
    """Runs workflows in the background; durable state lives in Postgres,
    live events and error text live in memory (lost on restart)."""

    def __init__(
        self,
        checkpointer,
        agents_factory: AgentsFactory | None = None,
        *,
        qa_client=None,
        qa_retriever=None,
        qa_factory: Callable[[], object] | None = None,
    ):
        self.checkpointer = checkpointer
        self.agents_factory = agents_factory
        self.qa_client = qa_client
        self.qa_retriever = qa_retriever
        self.qa_factory = qa_factory
        self.records: dict[str, RunRecord] = {}
        self._client = None  # lazily created, shared by all runs
        # Read-only graph for state lookups; its nodes are never executed.
        self._reader = build_workflow(
            agents=Agents(None, None, None), checkpointer=checkpointer
        )
        self._qa_reader = build_qa_workflow(
            client=_ReaderClient(), retriever=_reader_retriever,
            checkpointer=checkpointer,
        )

    # ---- internals -------------------------------------------------------

    def _default_factory(self, guardrails: GuardrailsInterceptor) -> Agents:
        if self._client is None:
            load_dotenv()
            self._client = from_env()
        return default_agents(guardrails, self._client)

    def _graph(self, graph_name: str = "code"):
        guardrails = GuardrailsInterceptor()
        if graph_name == "code":
            factory = self.agents_factory or self._default_factory
            return build_workflow(factory(guardrails), self.checkpointer, guardrails)
        if graph_name == "qa":
            if self.qa_factory is not None:
                return self.qa_factory()
            if self.qa_client is None or self.qa_retriever is None:
                raise RuntimeError("QA graph requires RAG to be enabled")
            return build_qa_workflow(
                self.qa_client, self.qa_retriever, self.checkpointer, guardrails,
            )
        raise ValueError(f"unknown graph: {graph_name}")

    @staticmethod
    def _graph_name(run_id: str) -> str:
        if run_id.startswith("qa-"):
            return "qa"
        if run_id.startswith("code-"):
            return "code"
        return "code"  # backwards-compatible IDs created before graph selection

    def _record(self, run_id: str) -> RunRecord:
        rec = self.records.get(run_id)
        if rec is None:
            rec = RunRecord(run_id, graph_name=self._graph_name(run_id))
            self.records[run_id] = rec
        return rec

    async def _emit(self, rec: RunRecord, event: dict):
        async with rec.cond:
            rec.events.append({"seq": len(rec.events), **event})
            rec.cond.notify_all()

    async def _run(self, rec: RunRecord, graph_input):
        config = {"configurable": {"thread_id": rec.run_id}}
        try:
            graph = self._graph(rec.graph_name)
            async for update in graph.astream(graph_input, config, stream_mode="updates"):
                for node, delta in update.items():
                    await self._emit(rec, {"event": "node", "node": node, "update": delta})
            snap = await graph.aget_state(config)
            interrupt_info = self._interrupt_info(snap)
            if interrupt_info is not None:
                rec.awaiting_approval = True
                await self._emit(rec, {"event": "awaiting_approval", "approval": interrupt_info})
            else:
                rec.awaiting_approval = False
                await self._emit(rec, {"event": "done"})
        except Exception as e:
            rec.error = f"{type(e).__name__}: {e}"
            await self._emit(rec, {"event": "error", "detail": rec.error})
        finally:
            async with rec.cond:
                rec.finished = True
                rec.cond.notify_all()

    def _launch(self, rec: RunRecord, graph_input):
        rec.finished, rec.error, rec.awaiting_approval = False, None, False
        rec.task = asyncio.create_task(self._run(rec, graph_input))

    def can_run(self, graph_name: str) -> bool:
        return graph_name == "code" or (
            graph_name == "qa" and (
                self.qa_factory is not None
                or (self.qa_client is not None and self.qa_retriever is not None)
            )
        )

    # ---- public API ------------------------------------------------------

    def start(self, task: str, graph: str = "code") -> str:
        if graph not in {"code", "qa"}:
            raise ValueError("graph must be 'code' or 'qa'")
        run_id = f"{graph}-{uuid.uuid4()}"
        rec = self.records[run_id] = RunRecord(run_id, graph_name=graph)
        if graph == "code":
            graph_input = {"task": task, "recursion_count": 0, "tokens_used": 0}
        else:
            graph_input = {
                "question": task, "tokens_used": 0, "attempts": 0,
                "revision_count": 0, "branch_results": [],
            }
        self._launch(rec, graph_input)
        return run_id

    def resume(self, run_id: str):
        rec = self._record(run_id)
        resume_value = Command(resume={"approve": True}) if rec.awaiting_approval else None
        self._launch(rec, resume_value)

    async def approve(self, run_id: str, approve: bool, feedback: str = ""):
        rec = self._record(run_id)
        info = await self.status(run_id)
        if info is None or info["status"] != "awaiting_approval":
            raise ValueError("run is not awaiting approval")
        self._launch(rec, Command(resume={"approve": approve, "feedback": feedback}))

    async def status(self, run_id: str) -> dict | None:
        rec = self.records.get(run_id)
        graph_name = rec.graph_name if rec else self._graph_name(run_id)
        reader = self._qa_reader if graph_name == "qa" else self._reader
        snap = await reader.aget_state({"configurable": {"thread_id": run_id}})
        v = snap.values or {}
        if not v and rec is None:
            return None
        interrupt_info = self._interrupt_info(snap)
        if (rec and rec.awaiting_approval) or interrupt_info is not None:
            status = "awaiting_approval"
        elif rec and rec.running:
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
            "attempts": v.get("attempts", v.get("recursion_count", 0)),
            "tokens_used": v.get("tokens_used", 0),
            "graph": graph_name,
            "question": v.get("question"),
            "route": v.get("route"),
            "sub_questions": v.get("sub_questions", []),
            "final_answer": v.get("final_answer"),
            "verdict": v.get("verdict"),
            "revision_count": v.get("revision_count", 0),
            "approval": interrupt_info,
            "approval_feedback": v.get("approval_feedback", ""),
        }

    @staticmethod
    def _interrupt_info(snapshot) -> dict | None:
        for task in getattr(snapshot, "tasks", ()):
            interrupts = getattr(task, "interrupts", ())
            if interrupts:
                value = getattr(interrupts[0], "value", None)
                return value if isinstance(value, dict) else {"value": value}
        return None

    async def aclose(self):
        for rec in self.records.values():
            if rec.running:
                rec.task.cancel()
        if self._client is not None:
            await self._client.aclose()


class _ReaderClient:
    async def complete(self, *_args, **_kwargs):
        raise RuntimeError("read-only graph reader cannot execute")


async def _reader_retriever(_question: str):
    raise RuntimeError("read-only graph reader cannot retrieve")
