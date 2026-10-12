import copy
import json
import multiprocessing
import threading
from contextlib import ExitStack, contextmanager, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from scripts import pujiang_loopback_transport as m
from scripts import pujiang_mmlu_transport as stored
from scripts.prepare_pujiang_mmlu_source import GENERATION


@pytest.fixture
def case():
    request = {
        "model": m.FIXTURE_MODEL,
        "messages": [{"role": "user", "content": "synthetic local fixture only"}],
        **copy.deepcopy(GENERATION),
        "stream": False,
    }
    raw = json.dumps(
        request, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode()
    return {
        "case_id": "loopback-synthetic",
        "candidate_request": request,
        "prompt_sha256": "1" * 64,
        "prompt_tokens": 16,
        "max_tokens": request["max_tokens"],
        "ignore_eos": False,
        "request_sha256": stored._sha(raw),
        "rendered_prompt_sha256": "3" * 64,
    }


@contextmanager
def fixture_server(mode="success", body=None):
    state = SimpleNamespace(
        nonce="a" * 64,
        post_count=0,
        get_paths=[],
        requests=[],
        stop=threading.Event(),
    )
    if body is None:
        body = json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "answer is B"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 16,
                    "completion_tokens": 2,
                    "total_tokens": 18,
                },
            }
        ).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, status, raw, declared_length=None, **headers):
            self.send_response(status)
            length = len(raw) if declared_length is None else declared_length
            self.send_header("Content-Length", str(length))
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(raw)

        def do_GET(self):
            state.get_paths.append(self.path)
            raw = json.dumps(
                {
                    "schema_version": "aqkv-loopback-fixture/v1",
                    "nonce": "b" * 64 if mode == "wrong-nonce" else state.nonce,
                    "model_execution": 0
                    if mode == "nonboolean-identity"
                    else mode == "model-server",
                }
            ).encode()
            if mode == "bad-handshake":
                raw = b"not json"
            self.reply(
                302 if mode == "handshake-redirect" else 200,
                raw,
                declared_length=len(raw) + 1 if mode == "truncated-handshake" else None,
            )

        def do_POST(self):
            state.post_count += 1
            state.requests.append(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.path == m.REQUEST_PATH
            if mode == "disconnect":
                self.connection.close()
                return
            if mode == "stall":
                state.stop.wait(4)
                return
            if mode == "trickle":
                self.send_response(200)
                self.send_header("Content-Length", "100")
                self.end_headers()
                for _ in range(100):
                    if state.stop.wait(0.08):
                        break
                    try:
                        self.wfile.write(b"x")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
                return
            if mode == "redirect":
                self.reply(302, b"answer is B", Location="/_never_follow_redirect")
            elif mode == "http-error":
                self.reply(503, b"answer is B")
            elif mode == "truncated-response":
                self.reply(200, body, declared_length=len(body) + 1)
            else:
                self.reply(200, body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}
    )
    thread.start()
    try:
        yield state, server.server_address[1]
    finally:
        state.stop.set()
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive()


@pytest.fixture
def server_factory():
    with ExitStack() as stack:
        yield lambda **kwargs: stack.enter_context(fixture_server(**kwargs))


def capture(case, root, state, port, **kwargs):
    return m.capture_loopback_fixture(
        case["case_id"],
        case["candidate_request"],
        root,
        port=port,
        fixture_nonce=state.nonce,
        **kwargs,
    )


def normalize(case, root, row):
    receipt = {
        "schema_version": stored.SCHEMA,
        "arm": "B0",
        "evidence_kind": "development-fixture",
        "scorer_bundle_sha256": "2" * 64,
        "request_method": "POST",
        "request_path": m.REQUEST_PATH,
        "requests": [row],
    }
    return stored.normalize_stored_arm(
        receipt, [case], root, "B0", "2" * 64, "development-fixture"
    )


def test_capture_real_local_http_and_keep_all_claims_off(
    tmp_path, case, server_factory, monkeypatch
):
    state, port = server_factory()
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    root = tmp_path / "capture"
    row, audit = capture(case, root, state, port)
    assert state.get_paths == [m.HANDSHAKE_PATH] and state.post_count == 1
    assert state.requests == [stored.read_artifact(root, row["request"])]
    assert json.loads(state.requests[0]) == case["candidate_request"]
    assert row["attempts"] == 1 and row["result"]["kind"] == "http"
    assert audit["owned_worker_stopped"] and audit["owned_worker_exitcode"] == 0
    assert audit["owned_worker_pid"] not in {
        child.pid for child in multiprocessing.active_children()
    }
    for key in (
        "execution_authorized",
        "serving_tokenizer_verified",
        "task_quality_claim",
        "performance_claim",
        "proxy_environment_used",
        "redirects_followed",
    ):
        assert audit[key] is False
    assert audit["model_requests_sent"] == 0
    assert (root.stat().st_mode & 0o777) == 0o700
    assert all((path.stat().st_mode & 0o777) == 0o600 for path in root.iterdir())
    outcomes, summary = normalize(case, root, row)
    assert outcomes[0]["success"] and outcomes[0]["output"] == "answer is B"
    assert not summary["transport_verified"] and not summary["execution_authorized"]


@pytest.mark.parametrize(
    "mode",
    [
        "wrong-nonce",
        "model-server",
        "bad-handshake",
        "handshake-redirect",
        "truncated-handshake",
        "nonboolean-identity",
    ],
)
def test_handshake_rejects_without_any_post(tmp_path, case, server_factory, mode):
    state, port = server_factory(mode=mode)
    root = tmp_path / "capture"
    with pytest.raises(ValueError, match="handshake/worker failed"):
        capture(case, root, state, port)
    assert state.post_count == 0
    assert not (root / "stored-request.json").exists()
    audit = json.loads((root / "capture-audit.json").read_bytes())
    assert audit["owned_worker_stopped"]
    assert audit["worker_observation"]["post_attempts"] == 0


@pytest.mark.parametrize("mode", ["stall", "trickle"])
def test_total_deadline_cannot_be_reset_by_partial_bytes(
    tmp_path, case, server_factory, mode
):
    state, port = server_factory(mode=mode)
    root = tmp_path / "capture"
    row, audit = capture(case, root, state, port, deadline_seconds=0.8)
    assert state.post_count == 1
    assert row["result"] == {"kind": "timeout"}
    assert audit["elapsed_seconds"] < 2.0 and audit["owned_worker_stopped"]
    assert audit["owned_worker_pid"] not in {
        child.pid for child in multiprocessing.active_children()
    }
    outcomes, summary = normalize(case, root, row)
    assert summary["case_count"] == 1 and summary["failure_counts"] == {"timeout": 1}
    assert not outcomes[0]["success"]


@pytest.mark.parametrize(
    "mode,kind",
    [
        ("redirect", "http-error"),
        ("http-error", "http-error"),
        ("disconnect", "connection-error"),
        ("truncated-response", "connection-error"),
    ],
)
def test_one_post_no_retry_or_redirect_and_failure_kept(
    tmp_path, case, server_factory, mode, kind
):
    state, port = server_factory(mode=mode)
    root = tmp_path / "capture"
    row, audit = capture(case, root, state, port)
    assert state.post_count == 1 and state.get_paths == [m.HANDSHAKE_PATH]
    assert not audit["redirects_followed"]
    outcomes, summary = normalize(case, root, row)
    assert not outcomes[0]["success"] and outcomes[0]["output"] == ""
    assert summary["failure_counts"] == {kind: 1}


def test_oversized_body_retains_prefix_but_never_scores_it(
    tmp_path, case, server_factory
):
    state, port = server_factory(body=b"answer is B" + b"x" * 256)
    root = tmp_path / "capture"
    row, audit = capture(case, root, state, port, max_body_bytes=64)
    assert row["result"] == {"kind": "response-too-large"}
    partial = audit["worker_observation"]["partial_body"]
    assert partial["bytes"] == 64 and len(stored.read_artifact(root, partial)) == 64
    assert audit["worker_observation"]["partial_body_not_scored"]
    outcomes, summary = normalize(case, root, row)
    assert outcomes[0]["output"] == "" and not outcomes[0]["success"]
    assert summary["failure_counts"] == {"response-too-large": 1}


def test_exact_size_body_is_not_discarded(tmp_path, case, server_factory):
    raw = b"not-json-but-exact-size"
    state, port = server_factory(body=raw)
    root = tmp_path / "capture"
    row, _ = capture(case, root, state, port, max_body_bytes=len(raw))
    assert stored.read_artifact(root, row["result"]["body"]) == raw
    outcomes, summary = normalize(case, root, row)
    assert not outcomes[0]["success"]
    assert summary["failure_counts"] == {"invalid-json-response": 1}


def test_existing_outputs_never_overwritten(tmp_path, case, server_factory):
    state, port = server_factory()
    root = tmp_path / "capture"
    capture(case, root, state, port)
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    with pytest.raises(FileExistsError):
        capture(case, root, state, port)
    assert state.post_count == 1
    assert before == {path.name: path.read_bytes() for path in root.iterdir()}


@pytest.mark.parametrize(
    "field,value",
    [
        ("port", True),
        ("port", 0),
        ("port", 65536),
        ("fixture_nonce", "missing"),
        ("fixture_nonce", "A" * 64),
        ("deadline_seconds", True),
        ("deadline_seconds", 0.1),
        ("deadline_seconds", 11),
        ("deadline_seconds", float("nan")),
        ("deadline_seconds", float("inf")),
        ("max_body_bytes", True),
        ("max_body_bytes", 0),
        ("max_body_bytes", m.MAX_BYTES + 1),
    ],
)
def test_bad_limits_reject_before_output_or_network(tmp_path, case, field, value):
    kwargs = {"port": 1, "fixture_nonce": "a" * 64, field: value}
    root = tmp_path / "capture"
    with pytest.raises(ValueError):
        m.capture_loopback_fixture(
            case["case_id"], case["candidate_request"], root, **kwargs
        )
    assert not root.exists()


@pytest.mark.parametrize("mutation", ["model", "stream", "nan", "oversized-request"])
def test_nonfixture_requests_reject_before_network(tmp_path, case, mutation):
    request = copy.deepcopy(case["candidate_request"])
    if mutation == "model":
        request["model"] = "Qwen/Qwen3.5-35B-A3B"
    elif mutation == "stream":
        request["stream"] = True
    elif mutation == "nan":
        request["temperature"] = float("nan")
    else:
        request["messages"][0]["content"] = "x" * (m.MAX_BYTES + 1)
    root = tmp_path / "capture"
    with pytest.raises(ValueError):
        m.capture_loopback_fixture(
            "synthetic", request, root, port=1, fixture_nonce="a" * 64
        )
    assert not root.exists()


def test_no_posix_deadline_support_rejects_before_network(tmp_path, case, monkeypatch):
    monkeypatch.delattr(m.signal, "setitimer")
    root = tmp_path / "capture"
    with pytest.raises(ValueError, match="POSIX"):
        m.capture_loopback_fixture(
            "synthetic", case["candidate_request"], root, port=1, fixture_nonce="a" * 64
        )
    assert not root.exists()


def test_symlink_destination_rejected(tmp_path, case):
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "capture"
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        m.capture_loopback_fixture(
            "synthetic", case["candidate_request"], link, port=1, fixture_nonce="a" * 64
        )
    assert list(actual.iterdir()) == []


@pytest.mark.parametrize(
    "timed_out,raw",
    [
        (True, b'{"post_attempts":'),
        (True, b"[]"),
        (True, b"x" * 8193),
        (False, b'{"result":'),
        (False, b"{}"),
        (False, b'{"result":[],"observation":{}}'),
        (False, b'{"result":{"kind":"unknown"},"observation":{}}'),
        (False, b'{"result":{"kind":"http"},"observation":[]}'),
        (False, b'{"result":{"kind":"timeout"},"observation":{"post_attempts":true}}'),
    ],
)
def test_incomplete_worker_metadata_keeps_audit_and_blocks_scoring(
    tmp_path, case, monkeypatch, timed_out, raw
):
    # A deterministic process double covers interruption between metadata writes.
    root = tmp_path / "capture"
    state = SimpleNamespace(alive=timed_out, closed=False)

    class Process:
        pid = 12345
        exitcode = 0

        def start(self):
            name = "post-attempt.json" if timed_out else "worker-result.json"
            m._store(root, name, raw)

        def join(self, _timeout):
            pass

        def is_alive(self):
            return state.alive

        def terminate(self):
            state.alive = False
            self.exitcode = -15

        def close(self):
            state.closed = True

    monkeypatch.setattr(
        m.multiprocessing,
        "get_context",
        lambda _mode: SimpleNamespace(Process=lambda **_kw: Process()),
    )
    with pytest.raises(ValueError, match="handshake/worker failed"):
        m.capture_loopback_fixture(
            case["case_id"],
            case["candidate_request"],
            root,
            port=1,
            fixture_nonce="a" * 64,
        )
    audit = json.loads((root / "capture-audit.json").read_bytes())
    assert audit["owned_worker_stopped"] and state.closed and not state.alive
    assert not (root / "stored-request.json").exists()
    assert audit["model_requests_sent"] == 0 and not audit["performance_claim"]
    if b'post_attempts":true' not in raw:
        assert audit["metadata_error"] and audit["result_kind"] == "worker-error"


def test_worker_start_failure_keeps_audit_without_a_receipt(
    tmp_path, case, monkeypatch
):
    state = SimpleNamespace(closed=False)

    class Process:
        pid = exitcode = None

        def start(self):
            raise OSError("synthetic spawn failure")

        def close(self):
            state.closed = True

    monkeypatch.setattr(
        m.multiprocessing,
        "get_context",
        lambda _mode: SimpleNamespace(Process=lambda **_kw: Process()),
    )
    root = tmp_path / "capture"
    with pytest.raises(ValueError, match="handshake/worker failed"):
        m.capture_loopback_fixture(
            case["case_id"],
            case["candidate_request"],
            root,
            port=1,
            fixture_nonce="a" * 64,
        )
    audit = json.loads((root / "capture-audit.json").read_bytes())
    assert audit["lifecycle_error"] == "OSError: synthetic spawn failure"
    assert not audit["owned_worker_started"] and state.closed
    assert not (root / "stored-request.json").exists()
