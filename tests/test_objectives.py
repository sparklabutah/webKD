"""Exercise the production trainer loss methods on small CPU tensors."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from webkd.training.token_weights import weighted_lm_loss

torch = pytest.importorskip("torch")


@pytest.fixture
def trainer():
    # Load the exact methods from the integration without importing browser,
    # accelerator, or model-serving dependencies.
    source = Path(__file__).resolve().parents[1] / "integrations/llamafactory/train/sft/trainer.py"
    tree = ast.parse(source.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    names = {"_webkd_target_logprobs", "_webkd_kl_compute_loss"}
    methods = [
        node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    obj = type("LossHarness", (), {name: namespace[name] for name in names})()
    obj._webkd_kl_as_gamma = False
    obj._webkd_kl_gamma_col = True
    obj._webkd_kl_beta = 0.3
    obj.state = SimpleNamespace(global_step=1)
    return obj


def test_weighted_loss_is_scale_invariant():
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(2, 5, 7, generator=generator, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2, 3, 4], [-100, 3, 2, -100, -100]])
    mask = labels.ne(-100).float()
    full = weighted_lm_loss(logits, labels, mask)
    half = weighted_lm_loss(logits, labels, mask * 0.5)
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 7), labels[:, 1:].reshape(-1), ignore_index=-100
    )
    torch.testing.assert_close(full, expected)
    torch.testing.assert_close(full, half)
    full.backward()
    assert torch.isfinite(logits.grad).all()


def test_combined_anchor_matches_formula_and_keeps_reference_frozen(trainer):
    logits = torch.tensor([[[1.0, 0.0, 2.0], [0.0, 2.0, 1.0], [1.0, 1.0, 1.0]]], requires_grad=True)
    reference = torch.zeros_like(logits, requires_grad=True)
    trainer._get_webkd_kl_ref = lambda device: lambda **kwargs: {"logits": reference}
    labels = torch.tensor([[-100, 2, 1]])
    ce_weights = torch.tensor([[0.0, 0.2, 1.0]])
    gamma = torch.tensor([[0.0, 1.0, 0.0]])
    inputs = {
        "input_ids": torch.tensor([[0, 1, 2]]),
        "labels": labels,
        "token_weights": ce_weights,
        "kl_gamma": gamma,
    }
    loss = trainer._webkd_kl_compute_loss(lambda **kwargs: {"logits": logits}, inputs)
    logp = logits[:, :-1].log_softmax(-1).gather(-1, labels[:, 1:, None]).squeeze(-1)
    ref_logp = (
        reference[:, :-1].detach().log_softmax(-1).gather(-1, labels[:, 1:, None]).squeeze(-1)
    )
    difference = (ref_logp - logp).clamp(-20, 8)
    ce = -(logp * ce_weights[:, 1:]).sum() / ce_weights.sum()
    anchor = ((difference.exp() - difference - 1) * gamma[:, 1:]).sum()
    torch.testing.assert_close(loss, ce + 0.3 * anchor)
    loss.backward()
    assert reference.grad is None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[:, 1].abs().sum() > 0  # Action still learns through CE.


def test_equal_reference_has_zero_anchor(trainer):
    logits = torch.tensor([[[0.0, 1.0, 2.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0]]])
    trainer._get_webkd_kl_ref = lambda device: lambda **kwargs: {"logits": logits.clone()}
    labels = torch.tensor([[-100, 2, 1]])
    weights = torch.tensor([[0.0, 1.0, 1.0]])
    inputs = {
        "input_ids": torch.tensor([[0, 1, 2]]),
        "labels": labels,
        "token_weights": weights,
        "kl_gamma": weights,
    }
    combined = trainer._webkd_kl_compute_loss(lambda **kwargs: {"logits": logits}, inputs)
    torch.testing.assert_close(combined, weighted_lm_loss(logits, labels, weights))
