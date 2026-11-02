"""Check checkpoint, replay and media budgets without deleting existing results."""

import os
import shutil

from vla_visualizer.config import bounded_setting

MIB = 1024**2


class StorageBudgetError(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__(
            f"Storage preflight failed ({report['failed_stage']}): used {report['used_bytes'] / MIB:.2f} MiB / "
            f"total limit {report['limit_bytes'] / MIB:.2f} MiB, required for new job {report['required_bytes'] / MIB:.2f} MiB, "
            f"deficit {report['deficit_bytes'] / MIB:.2f} MiB. Existing artifacts are preserved."
        )


def budgets(kind):
    # Delta contains only 23,072 float32 parameters (~93 KiB); no duplicated base or optimizer.
    return {
        "checkpoint": 256 * 1024 if kind == "training" else 0,
        "replay": (32 if kind == "simulation" else 24) * MIB,
        "media": (64 if kind == "simulation" else 8) * MIB,
        "diagnostics": MIB,
    }


def category(path):
    if (
        "checkpoint" in path.parts
        or ".checkpoint.staging" in path.parts
        or path.suffix == ".safetensors"
    ):
        return "checkpoint"
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".mp4", ".gif", ".webm"}:
        return "media"
    if (
        path.name in {"replay.json", "result.json", "rollout.json", "training_visualization.json"}
        or "captures" in path.parts
    ):
        return "replay"
    return "diagnostics"


def usage(root):
    sizes = dict.fromkeys(("checkpoint", "replay", "media", "diagnostics"), 0)
    seen = set()
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        sizes[category(path.relative_to(root))] += stat.st_size
    return sizes


def preflight(root, config):
    root.mkdir(parents=True, exist_ok=True)
    per_category = usage(root)
    budget = budgets(config.kind)
    used = sum(per_category.values())
    required = sum(budget.values())
    limit = bounded_setting("VLA_TOTAL_MB", 4096, 64, 16384) * MIB
    disk = shutil.disk_usage(root)
    stats = os.statvfs(root)
    report = {
        "kind": config.kind,
        "used_bytes": used,
        "used_by_category": per_category,
        "budget_by_category": budget,
        "required_bytes": required,
        "limit_bytes": limit,
        "filesystem_free_bytes": disk.free,
        "filesystem_free_inodes": stats.f_favail,
        "user_quota": "unavailable; not inferred from disk free",
        "base_cache_included": False,
        "retention": "manual export/archive; never auto-delete existing data",
        "deficit_bytes": max(0, used + required - limit),
        "failed_stage": None,
    }
    if used + required > limit:
        report["failed_stage"] = "total_job_quota"
    elif disk.free < required + MIB:
        report.update(failed_stage="filesystem_space", deficit_bytes=required + MIB - disk.free)
    elif stats.f_favail < 32:
        report.update(failed_stage="filesystem_inodes", deficit_bytes=0)
    if report["failed_stage"]:
        raise StorageBudgetError(report)
    return report
