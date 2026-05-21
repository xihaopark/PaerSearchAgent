#!/usr/bin/env python3
"""Agent-visible submission validation for benchmark output contracts.

This validator only checks public deliverable contracts from the task registry.
It must not read hidden ground truth, reference values, thresholds, or scoring
tolerances.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


def _empty_result() -> dict[str, Any]:
    return {
        "visible_ok": True,
        "missing_files": [],
        "empty_files": [],
        "parse_errors": [],
        "schema_errors": [],
        "metric_family_errors": [],
        "checked_files": [],
    }


def _expected_outputs(task: dict[str, Any]) -> list[str]:
    visible = task.get("visible_validation") or {}
    files = visible.get("files") or {}
    if files:
        return [str(x) for x in files.keys()]
    ev = task.get("evaluation") or {}
    out = [str(x) for x in ev.get("required_outputs", []) or []]
    if not out and task.get("success_artifact_glob"):
        glob = str(task["success_artifact_glob"])
        if "*" not in glob and "?" not in glob:
            out.append(glob)
    return out


def _read_csv_header(path: Path, *, delimiter: str = ",") -> list[str]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=delimiter)
        return next(reader, [])


def _read_csv_rows(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _has_metric_family(metric_names: list[str], aliases: list[str]) -> bool:
    normalized = [_norm(x) for x in metric_names]
    for alias in aliases:
        parts = [_norm(x) for x in re.split(r"[|,;/ ]+", alias) if x.strip()]
        if parts and any(all(part in name for part in parts) for name in normalized):
            return True
    return False


def validate_visible_submission(task: dict[str, Any], work_dir: str | Path) -> dict[str, Any]:
    """Validate public output contract for a task workspace.

    Args:
        task: Registry task entry. `visible_validation` is optional.
        work_dir: Agent workspace root.

    Returns:
        JSON-serializable dict. `visible_ok` is false if any visible check fails.
    """
    root = Path(work_dir)
    result = _empty_result()
    visible = task.get("visible_validation") or {}
    file_specs = visible.get("files") or {}

    expected = _expected_outputs(task)
    for rel in expected:
        rel = str(rel)
        path = root / rel
        spec = file_specs.get(rel, {})
        checked: dict[str, Any] = {"path": rel, "exists": path.is_file(), "nonempty": False}
        if not path.is_file():
            result["missing_files"].append(rel)
            result["checked_files"].append(checked)
            continue
        size = path.stat().st_size
        checked["size"] = size
        checked["nonempty"] = size > 0
        if size <= 0:
            result["empty_files"].append(rel)
            result["checked_files"].append(checked)
            continue

        ftype = str(spec.get("type") or "").lower()
        if not ftype:
            if rel.endswith(".csv"):
                ftype = "csv"
            elif rel.endswith(".tsv"):
                ftype = "tsv"
            elif rel.endswith(".json"):
                ftype = "json"

        if ftype in {"csv", "tsv"}:
            delimiter = "\t" if ftype == "tsv" else ","
            try:
                header = _read_csv_header(path, delimiter=delimiter)
                checked["columns"] = header
            except Exception as exc:
                result["parse_errors"].append({"path": rel, "error": f"{ftype}_parse_error:{exc}"})
                result["checked_files"].append(checked)
                continue
            required_columns = [str(x) for x in spec.get("required_columns", []) or []]
            missing_cols = [col for col in required_columns if col not in header]
            if missing_cols:
                result["schema_errors"].append({"path": rel, "missing_columns": missing_cols})

            metric_families = spec.get("metric_families") or {}
            min_unique = spec.get("min_unique_values") or {}
            if min_unique:
                try:
                    rows = _read_csv_rows(path, delimiter=delimiter)
                    col = str(min_unique.get("column") or "")
                    min_count = int(min_unique.get("min") or 0)
                    values = {str(row.get(col, "")).strip() for row in rows if str(row.get(col, "")).strip()}
                    if col and min_count and len(values) < min_count:
                        result["schema_errors"].append(
                            {
                                "path": rel,
                                "column": col,
                                "unique_values": len(values),
                                "min_unique_values": min_count,
                            }
                        )
                except Exception as exc:
                    result["parse_errors"].append({"path": rel, "error": f"csv_unique_error:{exc}"})
            if metric_families:
                try:
                    rows = _read_csv_rows(path)
                    metric_col = str(spec.get("metric_name_column") or "metric_name")
                    metric_names = [str(row.get(metric_col, "")) for row in rows]
                    min_rows = int(spec.get("min_rows") or 0)
                    if min_rows and len(rows) < min_rows:
                        result["schema_errors"].append(
                            {"path": rel, "row_count": len(rows), "min_rows": min_rows}
                        )
                    for family, aliases in metric_families.items():
                        alias_list = aliases if isinstance(aliases, list) else [aliases]
                        if not _has_metric_family(metric_names, [str(x) for x in alias_list]):
                            result["metric_family_errors"].append(
                                {"path": rel, "missing_metric_family": str(family)}
                            )
                except Exception as exc:
                    result["parse_errors"].append({"path": rel, "error": f"csv_rows_error:{exc}"})

        elif ftype == "json":
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except Exception as exc:
                result["parse_errors"].append({"path": rel, "error": f"json_parse_error:{exc}"})
                result["checked_files"].append(checked)
                continue
            checked["json_type"] = type(data).__name__
            required_keys = [str(x) for x in spec.get("required_keys", []) or []]
            if isinstance(data, dict):
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    result["schema_errors"].append({"path": rel, "missing_keys": missing_keys})
            elif required_keys:
                result["schema_errors"].append({"path": rel, "error": "json_not_object"})

        result["checked_files"].append(checked)

    result["visible_ok"] = not any(
        result[key]
        for key in ("missing_files", "empty_files", "parse_errors", "schema_errors", "metric_family_errors")
    )
    return result
