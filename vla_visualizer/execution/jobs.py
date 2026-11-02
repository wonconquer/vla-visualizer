"""Cross-process single-job lease, session ownership, bounded artifacts and status."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
import zipfile
from io import BytesIO

from vla_visualizer.config import ROOT, artifact_root, bounded_setting, worker_python

TERMINAL = {"completed", "failed", "cancelled", "timed_out", "resource_limit"}


def atomic_json(path: Path, value: dict) -> None:
    contents = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        temporary.write_text(contents + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def acquire_lease(root: Path | None = None):
    root = root or artifact_root()
    root.mkdir(parents=True, exist_ok=True)
    stream = (root / ".model-job.lock").open("a+")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RuntimeError(
            "Another model job is active. Wait for completion or cancel it before starting another."
        ) from None
    return stream


@contextmanager
def model_lease():
    # The supervisor already owns the lease for its one child.
    lease = None if os.environ.get("VLA_SUPERVISED") == "1" else acquire_lease()
    try:
        yield
    finally:
        if lease is not None:
            lease.close()


@dataclass(frozen=True)
class JobConfig:
    kind: str = "baseline"
    seed: int = 0
    device: str = "cpu"
    max_seconds: int = 300
    max_steps: int = 20
    action_steps: int = 10
    training_steps: int = 2
    batch_size: int = 1
    checkpoint_job: str | None = None
    instruction: str | None = None
    observation_mode: str = "canonical"
    episode_index: int = 0
    frame_index: int = 0

    def __post_init__(self):
        if (
            self.observation_mode not in {"canonical", "dataset"}
            or not 0 <= self.episode_index < 50
            or not 0 <= self.frame_index < 100_000
        ):
            raise ValueError("Invalid dataset observation selection.")
        from vla_visualizer.adapters.education_catalog import validate_instruction

        if self.kind != "simulation":
            validate_instruction(self.instruction, self.kind)
        elif self.instruction is not None:
            raise ValueError("The simulation instruction comes from the selected task definition.")
        if self.kind not in {"baseline", "simulation", "training", "evaluate"}:
            raise ValueError("Unsupported job kind.")
        if self.device not in {"cpu", "cuda"} or not 0 <= self.seed <= 2147483647:
            raise ValueError("Invalid device or seed.")
        if not 10 <= self.max_seconds <= bounded_setting("VLA_MAX_SECONDS", 900, 10, 1800):
            raise ValueError("Job wall-time configuration exceeds its bound.")
        if not 1 <= self.max_steps <= 280 or self.action_steps not in {1, 5, 10, 50}:
            raise ValueError("Invalid rollout/action chunk limit.")
        if not 1 <= self.training_steps <= 3 or self.batch_size != 1:
            raise ValueError(
                "Training smoke tests permit at most 3 optimizer steps with batch size 1."
            )
        if self.kind == "evaluate" and not self.checkpoint_job:
            raise ValueError("A parent training job ID is required for evaluation.")


class JobManager:
    def __init__(self, root: Path | None = None):
        self.root = (root or artifact_root()).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, job_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ValueError("Invalid job ID.")
        return self.root / job_id

    def _retention(self):
        path = self.root / ".record_retention.json"
        index = read_json(path) if path.exists() else {"schema_version": 1, "archived": {}}
        archived = index.get("archived")
        if index.get("schema_version") != 1 or not isinstance(archived, dict):
            raise ValueError("Invalid record-retention index. Existing artifacts are preserved.")
        if any(not re.fullmatch(r"[0-9a-f]{32}", key) for key in archived):
            raise ValueError("Invalid archived record ID. Existing artifacts are preserved.")
        return index

    def _archive_candidates(self, jobs, retention, recent_limit):
        recent = [
            j for j in jobs if j["kind"] != "evaluate" and j["run_id"] not in retention["archived"]
        ]
        needed = max(0, len(recent) + 1 - recent_limit)
        terminal = sorted(
            [j for j in recent if j["state"] in TERMINAL],
            key=lambda j: (j.get("finished_unix", j.get("created_unix", 0)), j["run_id"]),
        )
        if len(terminal) < needed:
            raise RuntimeError(
                f"Recent record window ({recent_limit}) contains too many unfinished jobs. "
                "Wait for completion or cancel the active job; no active record is archived."
            )
        return terminal[:needed]

    def status(self, job_id: str, owner: str) -> dict:
        directory = self.directory(job_id)
        status = read_json(directory / "status.json")
        if status["owner_hash"] != hashlib.sha256(owner.encode()).hexdigest():
            raise PermissionError("This job belongs to another session.")
        progress = directory / "progress.json"
        return {
            **status,
            "archived": job_id in self._retention()["archived"],
            "progress": read_json(progress) if progress.exists() else {},
        }

    def completed(self, owner, kind=None):
        owner_hash = hashlib.sha256(owner.encode()).hexdigest()
        records = [read_json(p) for p in self.root.glob("*/status.json")]
        return sorted(
            [
                r
                for r in records
                if r.get("owner_hash") == owner_hash
                and r["state"] == "completed"
                and (kind is None or r["kind"] == kind)
            ],
            key=lambda r: r.get("created_unix", 0),
            reverse=True,
        )

    def inventory(self):
        records = [read_json(p) for p in self.root.glob("*/status.json")]
        from collections import Counter

        primary = bounded_setting("VLA_MAX_JOBS", 20, 1, 100)
        archived = set(self._retention()["archived"]) & {j["run_id"] for j in records}
        return {
            "primary_limit": primary,
            "primary_limit_scope": "Recent primary records; terminal overflow is logically archived",
            "recent_primary_records": sum(
                j["kind"] != "evaluate" and j["run_id"] not in archived for j in records
            ),
            "retained_records": len(records),
            "retained_record_limit": bounded_setting("VLA_MAX_RECORDS", 200, 1, 2000),
            "evaluation_limit": bounded_setting("VLA_MAX_EVALUATIONS", 3 * primary, 1, 300),
            "evaluation_attempts_per_checkpoint": 3,
            "by_status": dict(Counter(r["state"] for r in records)),
            "by_kind": dict(Counter(r["kind"] for r in records)),
            "active": sum(r["state"] not in TERMINAL for r in records),
            "temporary_entries": [
                p.name
                for p in self.root.iterdir()
                if p.is_dir()
                and re.fullmatch(r"[0-9a-f]{32}", p.name)
                and not (p / "status.json").exists()
            ],
            "nested_status_records": sum(
                1 for p in self.root.rglob("status.json") if p.parent.parent != self.root
            ),
            "archived": len(archived),
            "archive_policy": (
                "Terminal primary overflow is logically archived. All paths, checkpoints, "
                "ownership and references remain accessible; all bytes and records still count. "
                "No automatic deletion."
            ),
        }

    def submit(self, config: JobConfig, owner: str) -> str:
        # Admission and deduplication are atomic independently of the long-lived model lease.
        with (self.root / ".admission.lock").open("a+") as admission:
            fcntl.flock(admission, fcntl.LOCK_EX)
            return self._submit_locked(config, owner)

    def _submit_locked(self, config, owner):
        jobs = [read_json(p) for p in self.root.glob("*/status.json")]
        primary_limit = bounded_setting("VLA_MAX_JOBS", 20, 1, 100)
        retention = self._retention()
        archive_candidates = []
        fingerprint = None
        if config.kind == "evaluate":
            parent = self.status(config.checkpoint_job, owner)
            if parent["kind"] != "training" or parent["state"] != "completed":
                raise ValueError("Select a completed training checkpoint owned by this session.")
            result = read_json(self.directory(config.checkpoint_job) / "result.json")
            request = {
                "checkpoint_job": config.checkpoint_job,
                "weight_sha256": result["saved_weight_sha256"],
                "seed": config.seed,
                "device": config.device,
                "evaluation_contract": 2,
            }
            fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
            children = [
                j
                for j in jobs
                if j["kind"] == "evaluate"
                and read_json(self.directory(j["run_id"]) / "config.json").get("checkpoint_job")
                == config.checkpoint_job
            ]
            for child in reversed(children):
                saved = read_json(self.directory(child["run_id"]) / "config.json")
                if saved.get("request_fingerprint") == fingerprint and (
                    child["state"] not in TERMINAL
                    or (
                        child["state"] == "completed"
                        and (self.directory(child["run_id"]) / "replay.json").exists()
                    )
                ):
                    self.status(child["run_id"], owner)
                    return child["run_id"]
            if len(children) >= 3:
                raise RuntimeError(
                    "This checkpoint has reached its bounded limit of 3 evaluation attempts. Existing results are preserved."
                )
            limit = bounded_setting("VLA_MAX_EVALUATIONS", 3 * primary_limit, 1, 300)
            if sum(j["kind"] == "evaluate" for j in jobs) >= limit:
                raise RuntimeError(
                    f"Evaluation record limit reached ({limit}). Checkpoints and existing evaluations are preserved."
                )
        record_limit = bounded_setting("VLA_MAX_RECORDS", 200, 1, 2000)
        if len(jobs) >= record_limit:
            raise RuntimeError(
                f"Retained record limit reached: {len(jobs)} / {record_limit}, including logical "
                "archives. Existing results are preserved. Export results and ask the operator "
                "to review retention capacity; logical archiving does not bypass this bound."
            )
        if config.kind != "evaluate":
            archive_candidates = self._archive_candidates(jobs, retention, primary_limit)
        lease = acquire_lease(self.root)
        directory = None
        retention_changed = False
        try:
            from vla_visualizer.execution.storage import preflight, StorageBudgetError

            try:
                storage = preflight(self.root, config)
            except StorageBudgetError as exc:
                failures = self.root / "submission_failures"
                failures.mkdir(exist_ok=True)
                atomic_json(failures / f"{uuid.uuid4().hex}.json", exc.report)
                raise
            payload = asdict(config)
            if fingerprint:
                payload["request_fingerprint"] = fingerprint
                payload["parent_training_run_id"] = config.checkpoint_job
                payload["checkpoint_sha256"] = request["weight_sha256"]
            payload["artifact_budget"] = storage["budget_by_category"]
            payload["storage_preflight"] = storage
            if config.checkpoint_job:
                source = self.status(config.checkpoint_job, owner)
                if source["kind"] != "training" or source["state"] != "completed":
                    raise ValueError("Only your completed training checkpoints can be evaluated.")
                payload["checkpoint_directory"] = str(
                    self.directory(config.checkpoint_job) / "checkpoint"
                )
            job_id = uuid.uuid4().hex
            directory = self.directory(job_id)
            directory.mkdir()
            atomic_json(directory / "config.json", payload)
            atomic_json(
                directory / "status.json",
                {
                    "run_id": job_id,
                    "owner_hash": hashlib.sha256(owner.encode()).hexdigest(),
                    "kind": config.kind,
                    "state": "starting",
                    "created_unix": time.time(),
                },
            )
            if archive_candidates:
                archived = dict(retention["archived"])
                for job in archive_candidates:
                    archived[job["run_id"]] = {
                        "archived_unix": time.time(),
                        "trigger_job": job_id,
                        "reason": "recent_primary_window",
                    }
                atomic_json(
                    self.root / ".record_retention.json",
                    {"schema_version": 1, "archived": archived},
                )
                retention_changed = True
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            # The inherited file description keeps the flock until supervision ends.
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "vla_visualizer.execution.supervisor",
                    str(directory),
                    str(lease.fileno()),
                    worker_python(config.kind),
                ],
                cwd=ROOT,
                env=env,
                pass_fds=(lease.fileno(),),
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return job_id
        except Exception:
            if retention_changed:
                atomic_json(self.root / ".record_retention.json", retention)
            if directory is not None:
                status = read_json(directory / "status.json")
                atomic_json(directory / "status.json", {**status, "state": "failed"})
            raise
        finally:
            lease.close()

    def cancel(self, job_id: str, owner: str) -> None:
        if self.status(job_id, owner)["state"] not in TERMINAL:
            (self.directory(job_id) / "cancel").touch()

    def export(self, job_id: str, owner: str) -> bytes:
        if self.status(job_id, owner)["state"] not in TERMINAL:
            raise ValueError("Only finished jobs can be exported.")
        directory = self.directory(job_id)
        files = [p for p in directory.rglob("*") if p.is_file() and not p.is_symlink()]
        # Full training checkpoints are downloadable separately via operator storage.
        delta = (directory / "checkpoint/checkpoint_manifest.json").exists()
        files = [p for p in files if delta or "checkpoint" not in p.relative_to(directory).parts]
        if sum(p.stat().st_size for p in files) > 64 * 1024**2:
            raise ValueError("Browser exports are limited to 64 MiB.")
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                if path.name != "status.json":
                    archive.write(path, path.relative_to(directory))
        return buffer.getvalue()
