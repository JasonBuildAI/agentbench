from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from .http_client import health_check
from .models import RunRecord
from .reporting import build_markdown_report, summarize_records, write_artifacts
from .utils import write_json


def run_benchmark(
    root: Path,
    suites: list[Any],
    base_url: str,
    provider_key: str,
    model: str,
    rounds: int,
    max_attempts: int,
    timeout_s: int,
) -> Path:
    health = health_check(base_url, timeout_s=5)
    if not health.ok:
        raise RuntimeError(f"OpenAgent health check failed: {health.error or health.text}")

    session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    session_dir = root / "results" / f"session-{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        session_dir / "meta.json",
        {
            "session_id": session_id,
            "base_url": base_url,
            "model": model,
            "rounds": rounds,
            "max_attempts": max_attempts,
            "timeout_s": timeout_s,
            "suites": [s.name for s in suites],
            "health": {"ok": health.ok, "status": health.status, "latency_ms": health.latency_ms},
        },
    )

    records: list[RunRecord] = []
    for suite in suites:
        suite_records = suite.run_suite(
            base_url=base_url,
            model=model,
            provider_key=provider_key,
            rounds=rounds,
            max_attempts=max_attempts,
            timeout_s=timeout_s,
        )
        records.extend(suite_records)

    summary = summarize_records(records)
    write_artifacts(
        session_dir,
        records,
        summary,
        session_id=session_id,
        base_url=base_url,
    )
    suite_names = sorted((summary.get("suites") or {}).keys())
    report = build_markdown_report(
        session_id,
        summary,
        base_url,
        suite_subdirs=suite_names,
    )
    (session_dir / "report.md").write_text(report, encoding="utf-8")
    return session_dir
