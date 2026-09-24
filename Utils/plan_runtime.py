"""Plan-level timing contract for the RobDP + Reqo experiment pipeline.

Operator timings remain in the plan tree. They are never used as a fallback
for a missing statement-level Execution Time.
"""

import json
import math
from pathlib import Path
from typing import Any

RUNTIME_METRIC = "execution_time"
RUNTIME_UNIT = "ms"


class RuntimeMetricError(ValueError):
    """The requested execution-time experiment cannot use these labels."""


def nonnegative_time(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeMetricError(f"Invalid {field}: {value!r}") from exc
    if not math.isfinite(result) or result < 0:
        raise RuntimeMetricError(f"Invalid {field}: {value!r}")
    return result


def cache_timing_fields(explain_doc: dict, analyze: bool) -> dict:
    """Keep statement timings alongside Plan when collecting a raw cache."""
    fields = {}
    for source, target in (("Execution Time", "execution_time_ms"),
                           ("Planning Time", "planning_time_ms")):
        if explain_doc.get(source) is not None:
            fields[target] = nonnegative_time(explain_doc[source], source)
    if analyze and "execution_time_ms" not in fields:
        raise RuntimeMetricError("EXPLAIN ANALYZE did not contain top-level Execution Time")
    return fields


def plan_timing_metadata(item: dict, require_label: bool = True) -> dict:
    """Keep both timings; use Execution Time (or an explicit timeout penalty)."""
    metadata = {"runtime_metric": RUNTIME_METRIC, "runtime_unit": RUNTIME_UNIT}
    root_time = item["plan"].get("Actual Total Time")
    if root_time is not None:
        metadata["root_actual_total_time_ms"] = nonnegative_time(root_time, "Actual Total Time")
    for field in ("execution_time_ms", "planning_time_ms"):
        if item.get(field) is not None:
            metadata[field] = nonnegative_time(item[field], field)

    timed_out = bool(item.get("timed_out")) or "timeout_runtime_ms" in item
    metadata["runtime_is_censored"] = timed_out
    if timed_out:
        penalty = nonnegative_time(item.get("timeout_runtime_ms"), "timeout_runtime_ms")
        if penalty <= 0 or "execution_time_ms" in metadata:
            raise RuntimeMetricError("Timeout requires a positive penalty and no measured Execution Time")
        metadata["timeout_runtime_ms"] = penalty
        metadata["runtime_label_ms"] = penalty
    elif "execution_time_ms" in metadata:
        metadata["runtime_label_ms"] = metadata["execution_time_ms"]
    elif require_label:
        key = tuple(item.get(k) for k in ("template_id", "original_query_id", "candidate_id"))
        raise RuntimeMetricError(
            f"Candidate {key} has no execution_time_ms. Legacy Plan-only caches cannot "
            "recover top-level Execution Time. Recollect the candidate cache using "
            "EXPLAIN ANALYZE; do not rename Actual Total Time."
        )
    return metadata


def execution_runtime_label(metadata: dict) -> float:
    """Reject legacy labels and mislabeled/corrupt encoded records."""
    if metadata.get("runtime_metric") != RUNTIME_METRIC or metadata.get("runtime_unit") != RUNTIME_UNIT:
        raise RuntimeMetricError("Expected execution_time labels in ms; re-encode this legacy dataset")
    field = "timeout_runtime_ms" if metadata.get("runtime_is_censored") else "execution_time_ms"
    value = nonnegative_time(metadata.get(field), field)
    label = nonnegative_time(metadata.get("runtime_label_ms"), "runtime_label_ms")
    if value != label:
        raise RuntimeMetricError(f"runtime_label_ms disagrees with {field}")
    if metadata.get("runtime_is_censored") and (value <= 0 or metadata.get("execution_time_ms") is not None):
        raise RuntimeMetricError("Invalid censored runtime label")
    return label


def require_execution_time_dataset(dataset_dir: Path, dbname: str) -> dict:
    """Preflight a converted dataset before starting GPU training."""
    path = dataset_dir / f"postgresql_{dbname}_executed_query_conversion_summary.json"
    if not path.is_file():
        raise RuntimeMetricError(f"Missing runtime manifest: {path}. Re-encode/reconvert the dataset.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("runtime_metric") != RUNTIME_METRIC or manifest.get("runtime_unit") != RUNTIME_UNIT:
        raise RuntimeMetricError(f"{path} is not an execution_time dataset in ms; regenerate it")
    return manifest
