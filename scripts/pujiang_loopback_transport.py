"""Exercise bounded HTTP against a nonce-bound local fixture, never a model."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import multiprocessing
import os
import re
import signal
import time
from pathlib import Path
from typing import Any

from scripts.validate_pujiang_mmlu_pair import _require
from scripts.validate_pujiang_readiness import parse_json

SCHEMA = "aqkv-loopback-transport-fixture/v1"
FIXTURE_MODEL = "aqkv-loopback-fixture"
HANDSHAKE_PATH = "/_aqkv_fixture"
REQUEST_PATH = "/v1/chat/completions"
MAX_BYTES = 1_048_576


def _store(root: Path, name: str, raw: bytes) -> dict[str, Any]:
    with (root / name).open("xb") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(raw)
    return {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _json(root: Path, name: str, value: object) -> None:
    _store(root, name, json.dumps(value, allow_nan=False).encode())


def _deadline(_signal: int, _frame: object) -> None:
    raise TimeoutError("Absolute fixture HTTP deadline exceeded")


def _worker(
    root: Path, port: int, nonce: str, request: bytes, end: float, limit: int
) -> None:
    connection = None
    observation: dict[str, Any] = {"phase": "handshake", "post_attempts": 0}
    result: dict[str, Any] = {"kind": "not-attempted"}
    try:
        remaining = end - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Deadline expired before fixture handshake")
        signal.signal(signal.SIGALRM, _deadline)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=remaining)
        connection.request("GET", HANDSHAKE_PATH)
        response = connection.getresponse()
        raw = response.read(1025)
        expected = {
            "schema_version": "aqkv-loopback-fixture/v1",
            "nonce": nonce,
            "model_execution": False,
        }
        identity = parse_json(raw)
        if (
            response.status != 200
            or len(raw) > 1024
            or response.length not in (None, 0)
            or identity != expected
            or identity.get("model_execution") is not False
        ):
            raise ValueError("Local fixture identity does not match the supplied nonce")
        connection.close()
        remaining = end - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Deadline expired before fixture POST")
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=remaining)
        # Record the initiated attempt; server receipt is separate evidence.
        observation.update(phase="post", post_attempts=1)
        _json(root, "post-attempt.json", observation)
        connection.request(
            "POST", REQUEST_PATH, request, {"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        observation["http_status"] = response.status
        raw = response.read(limit + 1)
        if len(raw) > limit:
            observation["partial_body"] = _store(
                root, "response-prefix.bin", raw[:limit]
            )
            observation["partial_body_not_scored"] = True
            result = {"kind": "response-too-large"}
        elif response.length not in (None, 0):
            observation["partial_body"] = _store(root, "response-prefix.bin", raw)
            observation["partial_body_not_scored"] = True
            raise http.client.IncompleteRead(raw, response.length)
        else:
            result = {
                "kind": "http",
                "status": response.status,
                "body": _store(root, "response.bin", raw),
            }
    except TimeoutError as error:
        result = {"kind": "timeout"}
        observation["error"] = str(error)
    except (OSError, http.client.HTTPException) as error:
        result = {"kind": "connection-error"}
        observation["error"] = f"{type(error).__name__}: {error}"[:1024]
    except Exception as error:
        result = {"kind": "worker-error"}
        observation["error"] = f"{type(error).__name__}: {error}"[:1024]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if connection is not None:
            connection.close()
    _json(root, "worker-result.json", {"result": result, "observation": observation})


def _stop_owned_worker(process: multiprocessing.Process) -> None:
    if process.is_alive():
        process.terminate()
        process.join(0.25)
    if process.is_alive():
        process.kill()
        process.join(1)
    _require(not process.is_alive(), "Owned fixture worker did not stop")


def _read_metadata(path: Path) -> dict:
    _require(not path.is_symlink(), "Worker metadata cannot be a symlink")
    with path.open("rb") as handle:
        raw = handle.read(8193)
    _require(len(raw) <= 8192, "Worker metadata exceeds the byte limit")
    value = parse_json(raw)
    _require(isinstance(value, dict), "Worker metadata must be an object")
    return value


def capture_loopback_fixture(
    case_id: str,
    request: dict,
    destination: Path,
    *,
    port: int,
    fixture_nonce: str,
    deadline_seconds: float = 2,
    max_body_bytes: int = MAX_BYTES,
) -> tuple[dict, dict]:
    """Capture one synthetic request in a new directory; there is no remote mode."""
    _require(
        isinstance(case_id, str) and 0 < len(case_id) <= 128,
        "Need a bounded fixture case ID",
    )
    _require(type(port) is int and 1 <= port <= 65535, "Invalid loopback port")
    _require(
        isinstance(fixture_nonce, str)
        and re.fullmatch(r"[0-9a-f]{64}", fixture_nonce) is not None,
        "Need the fixture server nonce",
    )
    _require(
        type(deadline_seconds) in (int, float)
        and math.isfinite(deadline_seconds)
        and 0.25 <= deadline_seconds <= 10,
        "Fixture total deadline must be between 0.25 and 10 seconds",
    )
    _require(
        type(max_body_bytes) is int and 1 <= max_body_bytes <= MAX_BYTES,
        "Invalid fixture response limit",
    )
    _require(
        isinstance(request, dict)
        and request.get("model") == FIXTURE_MODEL
        and request.get("stream") is False,
        "Only synthetic non-streaming fixture requests are allowed",
    )
    raw = json.dumps(
        request, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode()
    _require(len(raw) <= MAX_BYTES, "Fixture request exceeds the byte limit")
    _require(hasattr(signal, "setitimer"), "POSIX deadline support is required")
    _require(not destination.is_symlink(), "Fixture directory cannot be a symlink")
    destination.mkdir(mode=0o700)  # Existing outputs are never overwritten.
    destination = destination.resolve(strict=True)
    descriptor = _store(destination, "request.json", raw)
    context = multiprocessing.get_context("spawn")
    start = time.monotonic()
    process = context.Process(
        target=_worker,
        args=(
            destination,
            port,
            fixture_nonce,
            raw,
            start + deadline_seconds,
            max_body_bytes,
        ),
    )
    timed_out = False
    lifecycle_error = None
    try:
        process.start()
        process.join(max(0, deadline_seconds - (time.monotonic() - start)))
        timed_out = process.is_alive()
    except Exception as error:
        lifecycle_error = f"{type(error).__name__}: {error}"[:1024]
    finally:
        if process.pid is not None:
            _stop_owned_worker(process)
        pid, exitcode = process.pid, process.exitcode
        process.close()
    audit: dict[str, Any] = {
        "schema_version": SCHEMA,
        "evidence_kind": "development-fixture",
        "scope": "loopback-http-only-not-model-or-approved-serving",
        "deadline_seconds": deadline_seconds,
        "elapsed_seconds": time.monotonic() - start,
        "parent_deadline_triggered": timed_out,
        "owned_worker_started": pid is not None,
        "owned_worker_pid": pid,
        "owned_worker_exitcode": exitcode,
        "owned_worker_stopped": True,
        "maximum_post_attempts": 1,
        "redirects_followed": False,
        "proxy_environment_used": False,
        "max_body_bytes": max_body_bytes,
        "fixture_model": FIXTURE_MODEL,
        "server_received_post_verified_by_client": False,
        "execution_authorized": False,
        "serving_tokenizer_verified": False,
        "model_requests_sent": 0,
        "task_quality_claim": False,
        "performance_claim": False,
    }
    worker_file = destination / "worker-result.json"
    result, observation = {"kind": "worker-error"}, {}
    try:
        if lifecycle_error is not None:
            audit["lifecycle_error"] = lifecycle_error
        elif timed_out:
            attempt_file = destination / "post-attempt.json"
            observation = _read_metadata(attempt_file) if attempt_file.exists() else {}
            result = {"kind": "timeout"}
        elif exitcode == 0 and worker_file.is_file():
            completed = _read_metadata(worker_file)
            result, observation = completed["result"], completed["observation"]
        _require(
            isinstance(result, dict)
            and result.get("kind")
            in {
                "http",
                "timeout",
                "connection-error",
                "response-too-large",
                "worker-error",
            }
            and isinstance(observation, dict),
            "Invalid worker result structure",
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        result, observation = {"kind": "worker-error"}, {}
        audit["metadata_error"] = f"{type(error).__name__}: {error}"[:1024]
    audit["worker_observation"] = observation
    audit["result_kind"] = result["kind"]
    _json(destination, "capture-audit.json", audit)
    _require(
        type(observation.get("post_attempts")) is int
        and observation["post_attempts"] == 1
        and result["kind"] != "worker-error",
        "Fixture handshake/worker failed; no scorable transport receipt",
    )
    row = {
        "case_id": case_id,
        "request": descriptor,
        "attempts": 1,
        "result": result,
    }
    _json(destination, "stored-request.json", row)
    return row, audit
