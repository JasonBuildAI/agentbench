from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..benchcore.models import RunRecord


class SuiteBase(ABC):
    name: str

    @abstractmethod
    def load_tasks(self) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def run_task(
        self,
        task: dict[str, Any],
        round_index: int,
        attempt: int,
        base_url: str,
        model: str,
        provider_key: str,
        timeout_s: int,
    ) -> RunRecord:
        ...

    def run_suite(
        self,
        base_url: str,
        model: str,
        provider_key: str,
        rounds: int,
        max_attempts: int,
        timeout_s: int,
    ) -> list[RunRecord]:
        records: list[RunRecord] = []
        for task in self.load_tasks():
            for round_idx in range(1, rounds + 1):
                best: RunRecord | None = None
                for attempt in range(1, max_attempts + 1):
                    rec = self.run_task(
                        task=task,
                        round_index=round_idx,
                        attempt=attempt,
                        base_url=base_url,
                        model=model,
                        provider_key=provider_key,
                        timeout_s=timeout_s,
                    )
                    if rec.ok:
                        best = rec
                        break
                    if best is None:
                        best = rec
                if best is not None:
                    records.append(best)
                    print(
                        f"[{self.name}] {best.task_id} round {round_idx}/{rounds} "
                        f"ok={best.ok} latency_ms={best.latency_ms} attempt={best.attempt}"
                    )
        return records
