"""
provenance.py

What is needed to reproduce a run and to know where it ran: code version
(git commit, branch, uncommitted changes), package versions, host (local
machine or a batch cluster, with the job id), input data hashes, and
a small TOML writer for the human-readable run record (run.toml).
"""

import datetime
import hashlib
import math
import os
import platform
import subprocess
import sys
from importlib import metadata

from .config import CODE_ROOT

PACKAGES = ("numpy", "scipy", "pandas", "matplotlib", "pygad")


def _git(*arguments):
    try:
        result = subprocess.run(["git", *arguments], cwd=CODE_ROOT, capture_output=True,
                                text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def code_version():
    """Git commit, branch, remote, and whether tracked files have uncommitted changes."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "remote": _git("remote", "get-url", "origin"), "dirty": bool(status)}


def uncommitted_diff():
    """git diff of tracked files against HEAD (saved as code.patch when the tree is dirty)."""
    return _git("diff", "HEAD") or ""


def host_information(workers):
    """Machine, operating system, where the run ran, and the batch job if any."""
    hostname = platform.node()
    on_cluster = "SGE_O_HOST" in os.environ
    return {"hostname": hostname, "platform": platform.platform(),
            "where": "cluster" if on_cluster else "local",
            "job_id": os.environ.get("JOB_ID"), "task_id": os.environ.get("SGE_TASK_ID"),
            "slots": os.environ.get("NSLOTS"), "cpu_count": os.cpu_count(),
            "workers": workers}


def software_versions():
    versions = {"python": sys.version.split()[0]}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.datetime.now().astimezone()


# ─── TOML writer (the standard library reads TOML but does not write it) ──────


def _toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value if v is not None) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_toml_key(k)} = {_toml_value(v)}" for k, v in value.items()
                               if v is not None) + "}"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{text}"'


def _toml_key(key):
    key = str(key)
    return key if key.replace("_", "").replace("-", "").isalnum() else f'"{key}"'


def to_toml(data, prefix=""):
    """TOML text of nested dicts; lists of dicts become arrays of tables; None is omitted."""
    lines, tables, arrays = [], [], []
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, dict):
            tables.append((key, value))
        elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            arrays.append((key, value))
        else:
            lines.append(f"{_toml_key(key)} = {_toml_value(value)}")
    text = "\n".join(lines)
    for key, value in tables:
        name = f"{prefix}{_toml_key(key)}"
        body = to_toml(value, prefix=f"{name}.")
        text += f"\n\n[{name}]\n{body}" if body else f"\n\n[{name}]"
    for key, values in arrays:
        name = f"{prefix}{_toml_key(key)}"
        for value in values:
            text += f"\n\n[[{name}]]\n{to_toml(value, prefix=f'{name}.')}"
    return text.strip("\n")
