"""Provenance capture for every raw artifact.

Rule 2 of the workspace CLAUDE.md: every result carries provenance. The fields
below are the ones that file names, plus the router- and topology-specific ones
this project adds.

Two deliberate choices, both learned from P1:

* **Identifiers are copied, never retyped.** Model revision SHAs, image digests
  and backend versions are read out of P1's ``provision.lock.json`` and passed
  through verbatim. An agent previously fabricated a revision SHA here by
  autocompleting a truncated hash, which is exactly the class of error that a
  copy-from-ground-truth rule eliminates.
* **A missing value is absent, never a placeholder.** P1 shipped
  ``"version": "UNKNOWN"`` into a lockfile once and it propagated into real
  artifacts. Absent is honest; a placeholder looks like data.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOCKFILE_DEFAULT = Path(
    "/home/users/u104240/apps/inference-work/llm-inference-bench/provision.lock.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def git_sha(repo: Path) -> str | None:
    """Commit SHA, suffixed ``-dirty`` when the tree has uncommitted changes.

    The suffix is not cosmetic: an artifact produced from a dirty tree cannot be
    bit-reproduced from a clean checkout, and saying so is the difference
    between exact provenance and approximate provenance.

    Untracked files are ignored. The experiment writes its own artifacts into
    the repo, so counting them marked every phase after the first as dirty
    with no code change at all (job 5216059: stage-1 artifact clean, matrix
    artifacts "-dirty", same HEAD, only untracked results/ present).
    """
    sha = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    if sha is None:
        return None
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo)
    return f"{sha}-dirty" if status else sha


def gpu_info() -> dict[str, Any]:
    """GPU model, count, memory and driver, straight from nvidia-smi.

    Absent (not guessed) when nvidia-smi is unavailable, which is the case on
    the login node.
    """
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    if not out:
        return {}
    rows = [r.strip() for r in out.splitlines() if r.strip()]
    if not rows:
        return {}
    name, mem, driver = (p.strip() for p in rows[0].split(","))
    info: dict[str, Any] = {
        "gpu_model": name,
        "gpu_count": len(rows),
        "driver_version": driver,
    }
    if mem.endswith("MiB"):
        info["gpu_memory_gb"] = round(int(mem.removesuffix("MiB").strip()) / 1024, 1)
    cuda = _run(["nvidia-smi", "--query", "--display=COMPUTE"])
    if cuda:
        for line in cuda.splitlines():
            if "CUDA Version" in line:
                info["cuda_version"] = line.split(":")[-1].strip()
                break
    return info


def load_lockfile(path: Path = LOCKFILE_DEFAULT) -> dict[str, Any]:
    """Read P1's provisioning lockfile.

    Raises rather than returning defaults: a run that cannot state which image
    and which model revision it used must not produce an artifact at all.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"provisioning lockfile not found at {path}; "
            "cannot record backend version or model revision"
        )
    with path.open() as fh:
        return json.load(fh)


def backend_provenance(lock: dict[str, Any], backend: str, model_key: str) -> dict[str, Any]:
    """Pull backend and model identifiers out of the lockfile, verbatim.

    Every value here is copied. Nothing in this function constructs, truncates,
    completes or reformats an identifier.
    """
    image = lock.get("images", {}).get(backend)
    if not image:
        raise KeyError(f"lockfile has no image entry for backend {backend!r}")
    model = lock.get("models", {}).get(model_key)
    if not model:
        raise KeyError(f"lockfile has no model entry for {model_key!r}")

    out: dict[str, Any] = {
        "backend_name": backend,
        "model_id": model["repo"],
        "model_revision": model["revision"],
    }
    # Absent, not "UNKNOWN", when the lockfile lacks a field.
    for src, dst in (
        ("version", "backend_version"),
        ("digest", "backend_image_digest"),
        ("sif", "backend_image_sif"),
        ("pulled_utc", "backend_image_pulled_utc"),
    ):
        if image.get(src):
            out[dst] = image[src]
    if "backend_version" not in out:
        raise ValueError(
            f"lockfile has no version for backend {backend!r}; "
            "refusing to emit an artifact that cannot state its backend version"
        )
    return out


@dataclass
class RunProvenance:
    """The provenance block stamped onto every artifact."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    utc_timestamp: str = field(default_factory=utc_now)
    git_sha: str | None = None
    hostname: str = field(default_factory=platform.node)
    slurm_job_id: str | None = None
    exact_command_line: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def capture(cls, repo: Path, argv: list[str]) -> "RunProvenance":
        import shlex

        p = cls(
            git_sha=git_sha(repo),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            exact_command_line=shlex.join(argv),
        )
        p.extra.update(gpu_info())
        p.extra["slurm_nodelist"] = os.environ.get("SLURM_JOB_NODELIST")
        p.extra["python_version"] = platform.python_version()
        p.extra["kernel"] = platform.release()
        return p

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        extra = d.pop("extra")
        d.update({k: v for k, v in extra.items() if v is not None})
        return d
