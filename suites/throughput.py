from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Any

from ..benchcore.models import RunRecord
from ..benchcore.utils import now_ms
from .base import SuiteBase
from .easychat_common import run_chat_once


class ThroughputSuite(SuiteBase):
    name = "throughput"

    def __init__(self, root: Path) -> None:
        self.root = root

    def load_tasks(self) -> list[dict[str, Any]]:
        dataset_path = self.root / "datasets" / "throughput" / "dataset.jsonl"
        tasks: list[dict[str, Any]] = []
        if dataset_path.exists():
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        tasks.append(json.loads(line))
        return tasks

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
        res = run_chat_once(
            base_url=base_url,
            provider_key=provider_key,
            model=model,
            prompt=task["prompt"],
            timeout_s=timeout_s,
        )
        ok = bool(res["ok_http_parse"])
        return RunRecord(
            suite=self.name,
            task_id=task["id"],
            category=task.get("category", "throughput"),
            round_index=round_index,
            attempt=attempt,
            ok=ok,
            latency_ms=int(res["latency_ms"]),
            total_tokens=res["usage"].get("total_tokens"),
            prompt_tokens=res["usage"].get("prompt_tokens"),
            completion_tokens=res["usage"].get("completion_tokens"),
            response_text=res["assistant_text"],
            response_status=int(res["status"]),
            api_error=res["api_error"],
            parse_error=res["parse_error"],
            validation_errors=[],
            evidence={},
            extra={"throughput_mode": True},
        )

    def _run_concurrent_batch(
        self,
        task: dict[str, Any],
        round_index: int,
        base_url: str,
        model: str,
        provider_key: str,
        timeout_s: int,
    ) -> RunRecord:
        concurrency = int(task.get("concurrency", 1))
        requests_per_concurrency = int(task.get("requests_per_concurrency", 1))
        total_requests = concurrency * requests_per_concurrency

        overall_t0 = now_ms()
        sub_records: list[RunRecord] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    self.run_task, task, round_index, i + 1, base_url, model, provider_key, timeout_s
                )
                for i in range(total_requests)
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    sub_records.append(future.result())
                except Exception as exc:
                    sub_records.append(RunRecord(
                        suite=self.name,
                        task_id=task["id"],
                        category=task.get("category", "throughput"),
                        round_index=round_index,
                        attempt=0,
                        ok=False,
                        latency_ms=0,
                        total_tokens=None,
                        prompt_tokens=None,
                        completion_tokens=None,
                        response_text="",
                        response_status=0,
                        api_error=f"{type(exc).__name__}: {exc}",
                        parse_error=None,
                        validation_errors=["exception_in_future"],
                        evidence={},
                        extra={},
                    ))

        overall_t1 = now_ms()

        ok_records = [r for r in sub_records if r.ok]
        ok_count = len(ok_records)
        latencies = [r.latency_ms for r in ok_records]
        tokens = [r.total_tokens for r in ok_records if r.total_tokens is not None]

        total_time_ms = overall_t1 - overall_t0
        throughput_rps = (ok_count / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0

        validation_errors: list[str] = []
        if ok_count < total_requests:
            validation_errors.append(f"partial_failure:{ok_count}/{total_requests}")

        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        avg_tokens = sum(tokens) / len(tokens) if tokens else 0

        return RunRecord(
            suite=self.name,
            task_id=task["id"],
            category=task.get("category", "throughput"),
            round_index=round_index,
            attempt=1,
            ok=(ok_count == total_requests),
            latency_ms=int(avg_latency),
            total_tokens=int(avg_tokens) if avg_tokens else None,
            prompt_tokens=None,
            completion_tokens=None,
            response_text=ok_records[0].response_text if ok_records else "",
            response_status=200 if ok_count == total_requests else 500,
            api_error=None,
            parse_error=None,
            validation_errors=validation_errors,
            evidence={
                "concurrency": concurrency,
                "total_requests": total_requests,
                "successful_requests": ok_count,
                "total_time_ms": total_time_ms,
                "throughput_rps": round(throughput_rps, 2),
                "avg_latency_ms": round(avg_latency, 2),
                "min_latency_ms": min(latencies) if latencies else 0,
                "max_latency_ms": max(latencies) if latencies else 0,
                "latencies_ms": [r.latency_ms for r in sub_records],
            },
            extra={"throughput_mode": True},
        )

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
                rec = self._run_concurrent_batch(
                    task=task,
                    round_index=round_idx,
                    base_url=base_url,
                    model=model,
                    provider_key=provider_key,
                    timeout_s=timeout_s,
                )
                records.append(rec)
                print(
                    f"[{self.name}] {rec.task_id} round {round_idx}/{rounds} "
                    f"ok={rec.ok} latency_ms={rec.latency_ms} "
                    f"throughput_rps={rec.evidence.get('throughput_rps')}"
                )
        return records
