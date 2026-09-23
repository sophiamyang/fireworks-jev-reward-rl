"""Local first, atomic artifacts, no implicit replay of any RPC."""

import errno
import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import fields, is_dataclass
from pathlib import Path


def sync_dir(path):
    """Make a rename or creation in path durable; skip where directories cannot be fsynced."""
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EBADF):
            raise
    finally:
        os.close(fd)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, temp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        sync_dir(path.parent)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_error(exc):
    message = str(exc)
    for name, secret in os.environ.items():
        if secret and any(s in name.upper() for s in ("KEY", "TOKEN", "PASSWORD", "SECRET")):
            message = message.replace(secret, "[REDACTED]")
    message = re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", message)
    return {"error_type": type(exc).__name__, "message": message[:3000], "automatic_retry": False}


def fresh(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def rpc_json(value):
    """Lossless public SDK response data; reject unknown types, never stringify."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        # JSON has no NaN/inf; keep the receipt of a paid call rather than fail after it.
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise TypeError("RPC response mapping keys must be strings")
        return {k: rpc_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rpc_json(v) for v in value]
    # Tinker 0.23 returns dataclasses for backward results, Pydantic for saves.
    # TensorData must use its public flattened data, not its private ndarray.
    from tinker.types.tensor_data import TensorData

    if isinstance(value, TensorData):
        return {
            k: rpc_json(getattr(value, k))
            for k in ("data", "dtype", "shape", "sparse_crow_indices", "sparse_col_indices")
        }
    if hasattr(value, "model_dump"):
        return rpc_json(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: rpc_json(getattr(value, f.name)) for f in fields(value)}
    raise TypeError(f"Unsupported RPC response type: {type(value).__name__}")


def rpc(folder, name, call):
    """Journal BEFORE dispatch. Never replay an attempted operation, even after timeout."""
    attempt = Path(folder) / f"{name}-attempt.json"
    attempt.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents two local processes dispatching the same operation.
    with attempt.open("x") as f:
        json.dump({"started_unix": time.time(), "no_automatic_replay": True}, f)
        f.flush()
        os.fsync(f.fileno())
    sync_dir(attempt.parent)
    try:
        value = call()
        result = value.result(timeout=600) if hasattr(value, "result") else value
        save(Path(folder) / f"{name}-result.json", rpc_json(result))
        return result
    except BaseException as exc:
        save(Path(folder) / f"{name}-error.json", safe_error(exc))
        raise


class Budget:
    def __init__(self, folder, limits):
        self.path = Path(folder) / "budget.json"
        self.limits = dict(limits)
        self.attempted = {k: 0 for k in limits}
        self.write()

    def write(self):
        save(self.path, {"limits": self.limits, "attempted": self.attempted})

    def take(self, key, count=1):
        if not isinstance(count, int) or count < 1 or self.attempted[key] + count > self.limits[key]:
            raise ValueError(f"Request budget exceeded: {key}")
        self.attempted[key] += count
        self.write()
