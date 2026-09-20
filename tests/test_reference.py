"""CPU component tests; no runtime, graph or performance acceptance here."""

import pytest

torch = pytest.importorskip("torch")

from vllm_ascend_adaptive_quantized_kv import reference  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_case(
    *,
    kv_lengths=(48,),
    query_lengths=(7,),
    heads=4,
    kv_heads=2,
    dim=32,
    block_size=32,
    dtype=torch.float32,
    seed=19,
):
    generator = torch.Generator().manual_seed(seed)
    columns = (max(kv_lengths) + block_size - 1) // block_size
    pages = columns + 2
    shape = (pages, kv_heads, dim // 32, block_size, 32)
    return {
        "query": torch.randn(
            sum(query_lengths), heads, dim, generator=generator, dtype=dtype
        ),
        "key_nz": torch.randint(-8, 9, shape, generator=generator, dtype=torch.int8),
        "value_nz": torch.randint(-8, 9, shape, generator=generator, dtype=torch.int8),
        "block_table": torch.tensor(
            [list(range(columns - 1, -1, -1)) + [-1, 999] for _ in kv_lengths]
        ),
        "kv_lengths": list(kv_lengths),
        "query_lengths": list(query_lengths),
        "key_scale": torch.rand(kv_heads, dim, generator=generator, dtype=dtype) + 0.1,
        "value_scale": torch.rand(kv_heads, dim, generator=generator, dtype=dtype)
        + 0.2,
        "query_tile_size": 3,
    }


def dense_oracle(case):
    """Gather the whole logical sequence in tests only, independently of helper."""
    query = case["query"]
    block_size = case["key_nz"].shape[3]
    heads, dim = query.shape[1:]
    kv_heads = case["key_nz"].shape[1]
    dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    results = []
    offset = 0
    for row, (length, qlength) in enumerate(
        zip(case["kv_lengths"], case["query_lengths"], strict=True)
    ):
        count = (length + block_size - 1) // block_size
        ids = case["block_table"][row, :count].long()
        dense = []
        for cache_name, scale_name in (
            ("key_nz", "key_scale"),
            ("value_nz", "value_scale"),
        ):
            gathered = case[cache_name].index_select(0, ids)
            logical = gathered.permute(0, 3, 1, 2, 4).reshape(-1, kv_heads, dim)
            decoded = logical[:length].to(dtype) * case[scale_name].to(dtype)
            dense.append(decoded.repeat_interleave(heads // kv_heads, dim=1))
        q = query[offset : offset + qlength].to(dtype).transpose(0, 1)
        k, v = (tensor.transpose(0, 1) for tensor in dense)
        logits = torch.matmul(q, k.transpose(1, 2))
        logits *= case.get("softmax_scale", dim**-0.5)
        causal = (
            torch.arange(length)[None, :]
            <= torch.arange(length - qlength, length)[:, None]
        )
        weights = torch.softmax(logits.masked_fill(~causal, -torch.inf), dim=-1)
        results.append(torch.matmul(weights, v).transpose(0, 1).to(query.dtype))
        offset += qlength
    return torch.cat(results)


@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("heads,kv_heads,dim", [(2, 2, 32), (4, 2, 64), (4, 1, 32)])
@pytest.mark.parametrize("lengths,qlengths", [((48,), (7,)), ((35, 17), (35, 5))])
def test_matches_dense(dtype, heads, kv_heads, dim, lengths, qlengths):
    case = make_case(
        dtype=dtype,
        heads=heads,
        kv_heads=kv_heads,
        dim=dim,
        kv_lengths=lengths,
        query_lengths=qlengths,
    )
    actual = reference.paged_int8_attention_reference(**case)
    tolerance = {
        torch.float64: 1e-12,
        torch.float32: 2e-5,
        torch.float16: 4e-3,
        torch.bfloat16: 3e-2,
    }[dtype]
    torch.testing.assert_close(
        actual, dense_oracle(case), rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize("tile", [1, 7, 64])
@pytest.mark.parametrize("block_size", [1, 16, 32])
def test_tile_size_unit_scales_and_custom_softmax(tile, block_size):
    case = make_case(kv_lengths=(33,), query_lengths=(9,), block_size=block_size)
    case.update(query_tile_size=tile, softmax_scale=0.25)
    case["key_scale"].fill_(1)
    case["value_scale"].fill_(1)
    case["block_table"] = case["block_table"].int()
    torch.testing.assert_close(
        reference.paged_int8_attention_reference(**case),
        dense_oracle(case),
        rtol=2e-5,
        atol=2e-5,
    )


def test_ignores_unused_pages_and_tail_padding():
    case = make_case()
    expected = reference.paged_int8_attention_reference(**case)
    for name in ("key_nz", "value_nz"):
        case[name][2:].fill_(127)
        case[name][0, :, :, 16:].fill_(-128)
    torch.testing.assert_close(
        reference.paged_int8_attention_reference(**case), expected, rtol=0, atol=0
    )


def test_read_only_inference_and_no_stale_state():
    case = make_case()
    case["query"].requires_grad_(True)
    snapshots = {
        name: value.clone()
        for name, value in case.items()
        if isinstance(value, torch.Tensor)
    }
    actual = reference.paged_int8_attention_reference(**case)
    assert not actual.requires_grad
    for name, snapshot in snapshots.items():
        torch.testing.assert_close(case[name], snapshot, rtol=0, atol=0)
    for seed in (21, 22):
        changed = make_case(seed=seed)
        torch.testing.assert_close(
            reference.paged_int8_attention_reference(**changed),
            dense_oracle(changed),
            rtol=2e-5,
            atol=2e-5,
        )


def test_page_work_and_query_tile_shapes(monkeypatch):
    case = make_case(dim=64, kv_lengths=(81,), query_lengths=(40,))
    calls = []
    dequantize = reference._dequantize_page
    einsum = torch.einsum

    def tracked_page(*args):
        result = dequantize(*args)
        calls.append(tuple(result.shape))
        assert result.shape[0] <= 32
        assert result.shape[1:] == (2, 64)
        return result

    def tracked_einsum(equation, *args):
        if equation == "qhgd,bhd->hgqb":
            assert args[0].shape[0] <= case["query_tile_size"]
            assert args[1].shape[0] <= 32
            assert args[1].shape[1] == 2
        return einsum(equation, *args)

    monkeypatch.setattr(reference, "_dequantize_page", tracked_page)
    monkeypatch.setattr(torch, "einsum", tracked_einsum)
    actual = reference.paged_int8_attention_reference(**case)
    assert len(calls) == 14 * 3 * 2
    assert (17, 2, 64) in calls
    torch.testing.assert_close(actual, dense_oracle(case), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "field,change,match",
    [
        ("key_nz", lambda t: t.float(), "INT8"),
        ("value_nz", lambda t: t.float(), "INT8"),
        ("key_nz", lambda t: t.flatten(1, 2), "5D NZ"),
        ("key_nz", lambda t: t[..., :16], "5D NZ"),
        ("value_nz", lambda t: t[:1], "shapes must match"),
        ("query", lambda t: t[:0], "nonempty"),
        ("query", lambda t: t.int(), "nonempty"),
        ("query", lambda t: t[..., :16], "head dimension"),
        ("query", lambda t: t[:, :3], "GQA"),
        ("query", lambda t: t.fill_(float("nan")), "query must be finite"),
        ("query", lambda t: t.to("meta"), "CPU"),
        ("query", lambda t: None, "Torch tensors"),
        ("block_table", lambda t: t.float(), "integer tensor"),
        ("block_table", lambda t: t[0], "2D"),
        ("block_table", lambda t: t[:0], "rows must align"),
        ("block_table", lambda t: t[:, :1], "does not cover"),
        ("block_table", lambda t: t.fill_(-1), "page IDs"),
        ("block_table", lambda t: t.fill_(99), "page IDs"),
        ("key_scale", lambda t: t.unsqueeze(0), "scales must be floating"),
        ("value_scale", lambda t: t.int(), "scales must be floating"),
        ("key_scale", lambda t: t.fill_(0), "finite and positive"),
        ("value_scale", lambda t: t.fill_(-1), "finite and positive"),
        ("key_scale", lambda t: t.fill_(float("inf")), "finite and positive"),
        ("value_scale", lambda t: t.fill_(float("nan")), "finite and positive"),
        ("kv_lengths", lambda t: [0], "positive Python integers"),
        ("query_lengths", lambda t: [True], "positive Python integers"),
        ("kv_lengths", lambda t: [48.0], "positive Python integers"),
        ("kv_lengths", lambda t: torch.tensor([48]), "lists or tuples"),
        ("kv_lengths", lambda t: [], "rows must align"),
        ("kv_lengths", lambda t: [3], "cannot exceed"),
        ("query_lengths", lambda t: [6], "cover exactly"),
        ("query_tile_size", lambda t: 0, "tile size"),
        ("query_tile_size", lambda t: True, "tile size"),
    ],
)
def test_rejects_invalid_contract(field, change, match):
    case = make_case()
    case[field] = change(case[field])
    with pytest.raises(ValueError, match=match):
        reference.paged_int8_attention_reference(**case)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), True, "1"])
def test_rejects_invalid_softmax_scale(scale):
    with pytest.raises(ValueError, match="softmax scale"):
        reference.paged_int8_attention_reference(**make_case(), softmax_scale=scale)


def test_numerically_stable_large_finite_logits():
    case = make_case(dtype=torch.float64)
    case["query"] *= 10000
    torch.testing.assert_close(
        reference.paged_int8_attention_reference(**case),
        dense_oracle(case),
        rtol=1e-11,
        atol=1e-11,
    )


def test_rejects_overflow_instead_of_returning_nan():
    case = make_case()
    case["query"].fill_(torch.finfo(torch.float32).max)
    case["key_nz"].fill_(127)
    with pytest.raises(ValueError, match="nonfinite reference output"):
        reference.paged_int8_attention_reference(**case)


def test_exact_archived_positive_shape():
    case = make_case()
    case["key_nz"] = case["key_nz"][:2]
    case["value_nz"] = case["value_nz"][:2]
    case["block_table"] = torch.tensor([[0, 1]])
    case["key_scale"].fill_(1)
    case["value_scale"].fill_(1)
    torch.testing.assert_close(
        reference.paged_int8_attention_reference(**case),
        dense_oracle(case),
        rtol=2e-5,
        atol=2e-5,
    )


def test_noncontiguous_inputs():
    case = make_case(dim=64)
    for name, value in case.items():
        if isinstance(value, torch.Tensor):
            case[name] = torch.stack((value, value), dim=-1)[..., 0]
            assert not case[name].is_contiguous()
    case["kv_lengths"] = tuple(case["kv_lengths"])
    case["query_lengths"] = tuple(case["query_lengths"])
    torch.testing.assert_close(
        reference.paged_int8_attention_reference(**case),
        dense_oracle(case),
        rtol=2e-5,
        atol=2e-5,
    )


def test_external_autocast_does_not_change_math_contract():
    case = make_case()
    expected = reference.paged_int8_attention_reference(**case)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = reference.paged_int8_attention_reference(**case)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_default_device_cannot_redirect_cpu_allocations():
    case = make_case()
    expected = reference.paged_int8_attention_reference(**case)
    with torch.device("meta"):
        actual = reference.paged_int8_attention_reference(**case)
    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_distinct_request_tables_and_repeated_page_aliases():
    case = make_case(dim=128, kv_lengths=(70, 49, 33), query_lengths=(19, 8, 1))
    case["block_table"] = torch.tensor(
        [[2, 0, 1, -1, 999], [3, 1, -1, -1, 999], [4, 4, -1, -1, 999]]
    )
    torch.testing.assert_close(
        reference.paged_int8_attention_reference(**case),
        dense_oracle(case),
        rtol=2e-5,
        atol=2e-5,
    )
