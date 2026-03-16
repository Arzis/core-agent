# core/logger.py
"""
增强版性能追踪 — 增加预算超支告警
"""
import time
import uuid
import functools
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("RAG")


@dataclass
class StepRecord:
    step_name: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    budget_ms: float = 0.0          # ★ 预算
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    error: Optional[str] = None

    def finish(self, **meta):
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.metadata.update(meta)

    @property
    def over_budget(self) -> bool:
        return self.budget_ms > 0 and self.duration_ms > self.budget_ms


@dataclass
class QueryTrace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    query: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    steps: List[StepRecord] = field(default_factory=list)
    total_duration_ms: float = 0.0
    budget_ms: float = 0.0           # ★ 总预算
    final_answer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: StepRecord):
        self.steps.append(step)

    @property
    def elapsed_so_far(self) -> float:
        return sum(s.duration_ms for s in self.steps)

    @property
    def remaining_budget(self) -> float:
        if self.budget_ms <= 0:
            return float("inf")
        return self.budget_ms - self.elapsed_so_far

    def finish(self):
        self.total_duration_ms = sum(s.duration_ms for s in self.steps)

    def print_report(self):
        table = Table(
            title=f"🔍 RAG Trace [{self.trace_id}]",
            box=box.ROUNDED, show_lines=True,
        )
        table.add_column("步骤", style="cyan", width=30)
        table.add_column("耗时(ms)", style="magenta", justify="right", width=10)
        table.add_column("预算(ms)", justify="right", width=10)
        table.add_column("占比", style="green", justify="right", width=8)
        table.add_column("状态", justify="center", width=6)
        table.add_column("详情", style="dim", max_width=40)

        for step in self.steps:
            pct = (step.duration_ms / self.total_duration_ms * 100
                   if self.total_duration_ms > 0 else 0)
            if step.status == "skipped":
                icon = "⏭️"
            elif step.over_budget:
                icon = "⚠️"
            elif step.status == "success":
                icon = "✅"
            else:
                icon = "❌"

            budget_str = f"{step.budget_ms:.0f}" if step.budget_ms > 0 else "-"
            duration_style = "bold red" if step.over_budget else ""
            meta_str = ", ".join(f"{k}={v}" for k, v in step.metadata.items())

            table.add_row(
                step.step_name,
                f"[{duration_style}]{step.duration_ms:.1f}[/]",
                budget_str,
                f"{pct:.1f}%",
                icon,
                meta_str[:40] if meta_str else "-",
            )

        # ★ 总计行, 标注是否超预算
        total_style = "bold red" if (self.budget_ms > 0 and self.total_duration_ms > self.budget_ms) else "bold green"
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[{total_style}]{self.total_duration_ms:.1f}[/]",
            f"{self.budget_ms:.0f}" if self.budget_ms > 0 else "-",
            "100%", "", "",
        )
        console.print(table)

    def to_dict(self) -> Dict:
        return {
            "trace_id": self.trace_id, "query": self.query,
            "total_ms": self.total_duration_ms,
            "budget_ms": self.budget_ms,
            "over_budget": self.budget_ms > 0 and self.total_duration_ms > self.budget_ms,
            "steps": [{"name": s.step_name, "ms": s.duration_ms, "budget": s.budget_ms} for s in self.steps],
        }


class PerformanceTracker:
    def __init__(self):
        self._current_trace: Optional[QueryTrace] = None
        self._history: List[QueryTrace] = []

    def new_trace(self, query: str, budget_ms: float = 0) -> QueryTrace:
        self._current_trace = QueryTrace(query=query, budget_ms=budget_ms)
        return self._current_trace

    @property
    def current(self) -> Optional[QueryTrace]:
        return self._current_trace

    @contextmanager
    def track_step(self, step_name: str, budget_ms: float = 0):
        step = StepRecord(step_name=step_name, start_time=time.perf_counter(), budget_ms=budget_ms)
        logger.info(f"[{self._current_trace.trace_id}] ▶ {step_name}")
        try:
            yield step
            step.finish()
            step.status = "success"
        except Exception as e:
            step.finish()
            step.status = "failed"
            step.error = str(e)
            logger.error(f"[{self._current_trace.trace_id}] ✖ {step_name}: {e}")
            raise
        finally:
            self._current_trace.add_step(step)
            budget_info = f" (预算:{budget_ms:.0f}ms)" if budget_ms > 0 else ""
            warn = " ⚠️超时!" if step.over_budget else ""
            logger.info(f"[{self._current_trace.trace_id}] ✔ {step_name} [{step.duration_ms:.1f}ms]{budget_info}{warn}")

    def add_skipped_step(self, step_name: str, reason: str = ""):
        step = StepRecord(step_name=step_name, start_time=time.perf_counter())
        step.end_time = step.start_time
        step.duration_ms = 0
        step.status = "skipped"
        step.metadata["skip_reason"] = reason
        self._current_trace.add_step(step)
        logger.info(f"[{self._current_trace.trace_id}] ⏭ {step_name} (跳过: {reason})")

    def finish_trace(self) -> QueryTrace:
        self._current_trace.finish()
        self._history.append(self._current_trace)
        return self._current_trace

    def get_avg_latency(self, last_n: int = 20) -> float:
        recent = self._history[-last_n:]
        if not recent:
            return 0
        return sum(t.total_duration_ms for t in recent) / len(recent)

    def print_summary(self):
        if not self._history:
            return
        table = Table(title="📊 查询历史", box=box.SIMPLE_HEAVY)
        table.add_column("ID", style="cyan", width=8)
        table.add_column("查询", max_width=35)
        table.add_column("耗时(ms)", style="magenta", justify="right")
        table.add_column("预算(ms)", justify="right")
        table.add_column("步骤", justify="center")
        table.add_column("状态", justify="center")

        for t in self._history[-15:]:
            over = t.budget_ms > 0 and t.total_duration_ms > t.budget_ms
            table.add_row(
                t.trace_id, t.query[:35],
                f"{t.total_duration_ms:.0f}",
                f"{t.budget_ms:.0f}" if t.budget_ms > 0 else "-",
                str(len([s for s in t.steps if s.status != "skipped"])),
                "⚠️" if over else "✅",
            )
        avg = self.get_avg_latency()
        console.print(table)
        console.print(f"  [dim]平均延迟: {avg:.0f}ms[/dim]")


def timed(step_name: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                ms = (time.perf_counter() - start) * 1000
                logger.info(f"⏱ [{step_name}] {ms:.1f}ms")
                return result
            except Exception as e:
                ms = (time.perf_counter() - start) * 1000
                logger.error(f"⏱ [{step_name}] FAIL {ms:.1f}ms: {e}")
                raise
        return wrapper
    return decorator


perf_tracker = PerformanceTracker()