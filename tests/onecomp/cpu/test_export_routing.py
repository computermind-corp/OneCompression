"""Unit tests for CPU/GGUF export routing, DBF dequant and rotation de-folding.

These run without any model download or llama.cpp build:
  * ``read_quant_meta`` / ``plan_export`` route each quant_method correctly,
  * the DBF dequantize matches ``DoubleBinaryLinear`` forward, and
  * the rotation Hadamard de-fold inverts the online transform exactly.

Copyright 2025-2026 Fujitsu Ltd.
"""

import json

import pytest
import torch


def _write_quant_config(tmp_path, quant_method, **extra):
    cfg = {"model_type": "llama", "quantization_config": {"quant_method": quant_method, **extra}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return str(tmp_path)


@pytest.mark.parametrize(
    "method,extra,expected_path,is_family",
    [
        ("gptq", {}, "direct", True),
        ("mixed_gptq", {}, "mixed", True),
        ("jointq", {}, "direct", True),
        ("rtn", {}, "direct", True),
        ("dbf", {}, "fallback", False),
        ("autobit", {}, "fallback", False),
        ("onebit", {}, "unsupported", False),
        # MDBF is rejected up-front; the rotated row pins guard-before-rotation.
        ("mdbf", {}, "unsupported", False),
        ("mdbf", {"rotated": True}, "unsupported", False),
        ("gptq", {"rotated": True}, "fallback", True),
        ("mixed_gptq", {"rotated": True}, "fallback", True),
        # act-order uniform GPTQ must go to mixed (direct packing isn't block-aligned)
        ("gptq", {"desc_act": True}, "mixed", True),
        ("gptq", {"actorder": True}, "mixed", True),
        # low-bit uniform GPTQ must go to mixed (no lossless GGUF block type)
        ("gptq", {"bits": 2}, "mixed", True),
        ("gptq", {"bits": 3}, "mixed", True),
        ("jointq", {"bits": 2}, "mixed", True),
        # rotation takes precedence over act-order routing
        ("gptq", {"desc_act": True, "rotated": True}, "fallback", True),
    ],
)
def test_plan_export_routing(tmp_path, method, extra, expected_path, is_family):
    from onecomp.cpu.export.auto import plan_export
    from onecomp.cpu.export.checkpoint import read_quant_meta

    d = _write_quant_config(tmp_path, method, **extra)
    meta = read_quant_meta(d)
    assert meta.quant_method == method
    assert meta.is_gptq_family is is_family
    assert meta.rotated is bool(extra.get("rotated", False))
    assert meta.actorder is bool(extra.get("desc_act", extra.get("actorder", False)))

    plan = plan_export(d)
    assert plan["path"] == expected_path


def test_plan_export_mixed_for_per_layer_low_bits(tmp_path):
    from onecomp.cpu.export.auto import plan_export

    qbits = [{"self_attn.q_proj": {"bits": 3, "method": "gptq", "params": {"group_size": 128}}}]
    d = _write_quant_config(tmp_path, "gptq", bits=4, quantization_bits=qbits)
    assert plan_export(d)["path"] == "mixed"


def test_needs_mixed_export_helpers():
    from onecomp.cpu.export.checkpoint import configured_bit_widths, needs_mixed_export

    assert not needs_mixed_export({"bits": 4, "sym": True})
    assert needs_mixed_export({"bits": 2, "sym": True})
    assert needs_mixed_export({"bits": 8, "sym": False})
    assert needs_mixed_export({"bits": 4, "desc_act": True})
    assert configured_bit_widths(
        {"bits": 4, "quantization_bits": [{}, {"mlp.down_proj": {"bits": 2}}]}
    ) == {4, 2}


@pytest.mark.parametrize("method", ["onebit", "mdbf"])
@pytest.mark.parametrize("mode", ["auto", "direct", "mixed", "fallback"])
def test_export_to_gguf_rejects_unsupported(tmp_path, method, mode):
    """An explicit ``mode`` names a path, not a capability: it must not bypass the guard."""
    from onecomp.cpu.export.auto import export_to_gguf

    d = _write_quant_config(tmp_path, method)
    with pytest.raises(ValueError, match="not supported"):
        export_to_gguf(d, str(tmp_path / "out.gguf"), mode=mode)


@pytest.mark.parametrize("method", ["onebit", "mdbf"])
def test_dequantize_to_hf_rejects_unsupported(tmp_path, method):
    """The low-level entry point guards too; it is public and reached via other paths."""
    from onecomp.cpu.export.dequantize import dequantize_to_hf

    d = _write_quant_config(tmp_path, method)
    with pytest.raises(ValueError, match="no dense reconstruction"):
        dequantize_to_hf(d, str(tmp_path / "dense"))


def test_reject_unfilled_weights_flags_random_init_tensors():
    """An unknown layout leaves dense weights unsourced; that must raise, not warn."""
    from onecomp.cpu.export.dequantize import _reject_unfilled_weights

    missing = [
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp.down_proj.bias",
        "model.rotary_emb.inv_freq",  # a buffer, legitimately absent
    ]
    with pytest.raises(RuntimeError, match="random init"):
        _reject_unfilled_weights(missing, set(), "/ckpt", "future_method")


def test_reject_unfilled_weights_ignores_buffers_and_retied_lm_head():
    from onecomp.cpu.export.dequantize import _reject_unfilled_weights

    _reject_unfilled_weights(["model.rotary_emb.inv_freq"], set(), "/ckpt", "gptq")
    _reject_unfilled_weights(
        ["lm_head.weight"], {"lm_head.weight"}, "/ckpt", "gptq"
    )  # restored by tie_weights()


def test_dequantize_to_hf_rejects_unknown_layout_end_to_end(tmp_path):
    """``UNSUPPORTED_METHODS`` is an allow-list of *known* gaps; this pins the net.

    A quant_method nobody listed (a future quantizer, or MDBF children hidden
    inside an ``autobit`` checkpoint) reaches the dequantize body, drops its
    tensors and leaves the dense weights at ``from_config`` random init. Only an
    end-to-end call proves ``_reject_unfilled_weights`` is actually wired into
    ``dequantize_to_hf``; the unit tests above pass even if the call is deleted.
    """
    from safetensors.torch import save_file
    from transformers import LlamaConfig

    from onecomp.cpu.export.dequantize import dequantize_to_hf

    config = LlamaConfig(
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        num_key_value_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
        vocab_size=32,
        tie_word_embeddings=False,
    )
    # Keep the output outside the checkpoint so the shard glob cannot see it.
    ckpt = tmp_path / "ckpt"
    out = tmp_path / "dense"
    ckpt.mkdir()

    cfg_dict = config.to_dict()
    cfg_dict["quantization_config"] = {"quant_method": "future_method", "bits": 2}
    (ckpt / "config.json").write_text(json.dumps(cfg_dict), encoding="utf-8")

    # A layer stored in some unknown factorized form: no ``.weight``, and keys
    # neither the GPTQ nor the DBF reader recognises.
    save_file(
        {
            "model.layers.0.self_attn.q_proj.factor_a": torch.zeros(16, 4),
            "model.layers.0.self_attn.q_proj.factor_b": torch.zeros(4, 16),
        },
        str(ckpt / "model.safetensors"),
    )

    with pytest.raises(RuntimeError, match="random init"):
        dequantize_to_hf(str(ckpt), str(out))
    assert not (out / "model.safetensors").exists(), "must not write a broken model"


def test_dbf_dequantize_matches_forward():
    """Identity-forward reconstruction reproduces DoubleBinaryLinear outputs."""
    from onecomp.cpu.export.dequantize import _dequantize_dbf_layers
    from onecomp.quantizer.dbf.dbf_layer import DoubleBinaryLinear, pack_binary

    torch.manual_seed(0)
    in_f, mid, out_f = 16, 24, 8

    def _rand_binary(rows, cols):
        return (torch.randint(0, 2, (rows, cols)) * 2 - 1).to(torch.float16)

    dbf_B = _rand_binary(mid, in_f)
    dbf_A = _rand_binary(out_f, mid)
    state = {
        "lin.scaling0": torch.randn(in_f).to(torch.float16),
        "lin.scaling2": torch.randn(mid).to(torch.float16),
        "lin.scaling4": torch.randn(out_f).to(torch.float16),
        "lin.bp1": pack_binary(dbf_B),
        "lin.bp3": pack_binary(dbf_A),
        "lin.bias": torch.randn(out_f).to(torch.float16),
    }

    # Reference layer.
    lsd = {k.split(".")[-1]: v for k, v in state.items()}
    ref = DoubleBinaryLinear.from_saved_state({k: v for k, v in lsd.items()}, in_f, out_f).eval()

    # A tiny dense model stub exposing the layer's in/out features.
    class _Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(in_f, out_f, bias=True)

    model = _Stub()
    dense, consumed = _dequantize_dbf_layers(model, state, torch.float32)

    assert "lin.weight" in dense and "lin.bias" in dense
    w = dense["lin.weight"]
    b = dense["lin.bias"]

    x = torch.randn(5, in_f, dtype=torch.float16)
    with torch.no_grad():
        expected = ref(x).float()
        got = (x.float() @ w.t()) + b.float()
    assert torch.allclose(expected, got, atol=1e-2, rtol=1e-2)


def test_hadamard_defold_roundtrip():
    """De-fold inverts the online down_proj Hadamard applied during rotation."""
    from onecomp.cpu.export.rotation import defold_down_proj_hadamard
    from onecomp.pre_process.hadamard_utils import get_hadK, matmul_hadU_cuda

    torch.manual_seed(1)
    out_f, in_f = 7, 64  # power-of-2 in_features

    w_orig = torch.randn(out_f, in_f, dtype=torch.float32)
    # Simulate rotate_down_proj's online Hadamard fold on the input dim.
    w_stored = matmul_hadU_cuda(w_orig, *get_hadK(in_f))
    # De-folding must recover the original (pre-Hadamard) weight.
    w_recovered = defold_down_proj_hadamard(w_stored)
    assert torch.allclose(w_orig, w_recovered, atol=1e-4)


def test_hadamard_defold_roundtrip_block():
    """Same round-trip for a non-power-of-2 dim using a Hadamard block kernel."""
    from onecomp.cpu.export.rotation import defold_down_proj_hadamard
    from onecomp.pre_process.hadamard_utils import get_hadK, matmul_hadU_cuda

    torch.manual_seed(2)
    out_f, in_f = 5, 12 * 8  # 96 = 12 * 2^3 -> uses the had12 block

    w_orig = torch.randn(out_f, in_f, dtype=torch.float32)
    w_stored = matmul_hadU_cuda(w_orig, *get_hadK(in_f))
    w_recovered = defold_down_proj_hadamard(w_stored)
    assert torch.allclose(w_orig, w_recovered, atol=1e-4)
