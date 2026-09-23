"""Optional CPU integration test using a random model and a synthetic tokenizer."""

import copy
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("WEBKD_TEST_INTEGRATION") != "1",
    reason="Set WEBKD_TEST_INTEGRATION=1 after installing the LLaMA-Factory integration",
)


def test_preprocessing_collation_and_combined_loss(tmp_path, monkeypatch):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments, FinetuningArguments, TrainingArguments
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

    from webkd import weights
    from webkd.training.token_weights import weighted_lm_loss

    torch.set_num_threads(1)
    torch.manual_seed(42)
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]"
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    data_args = DataArguments(template="qwen", cutoff_len=256)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    target = (
        "<think>\nLook at the visible search box.\n</think>\n\n"
        "<plan>\nSearch and read.\n</plan>\n\n<step>1</step>\n\n"
        "<memory>\nNo query entered.\n</memory>\n\n<action>\nclick('14')\n</action>"
    )
    ce_path, gamma_path = tmp_path / "ce.json", tmp_path / "gamma.json"
    weights.write_sidecar(ce_path, {target: weights.weight_spans_from_tags(target, {"think": 0.1})})
    weights.write_sidecar(
        gamma_path,
        {
            target: weights.weight_spans_from_tags(
                target, {"think": 1.0, "plan": 1.0, "memory": 1.0}
            )
        },
        default_weight=1e-6,
        floor=0.0,
    )
    monkeypatch.setenv("WEBKD_TOKEN_WEIGHTS", str(ce_path))
    monkeypatch.setenv("WEBKD_KL_GAMMA_WEIGHTS", str(gamma_path))
    monkeypatch.setenv("WEBKD_KL_REF", "synthetic-reference")
    monkeypatch.setenv("WEBKD_KL_BETA", "0.3")
    monkeypatch.delenv("WEBKD_KL_AS_GAMMA", raising=False)
    monkeypatch.delenv("WEBKD_TOKEN_WEIGHT_FLOOR", raising=False)
    examples = {
        "_prompt": [[{"role": "user", "content": "Find the page."}]],
        "_response": [[{"role": "assistant", "content": target}]],
        "_system": [""],
        "_tools": [""],
        "_images": [[]],
        "_videos": [[]],
        "_audios": [[]],
    }
    processor = SupervisedDatasetProcessor(
        template=template, tokenizer=tokenizer, processor=None, data_args=data_args
    )
    columns = processor.preprocess_dataset(examples)
    assert "token_weights" in columns and "kl_gamma" in columns
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model = LlamaForCausalLM(config)
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        tokenizer=tokenizer,
        model=model,
        processor=None,
        label_pad_token_id=-100,
        block_diag_attn=False,
        attn_implementation="eager",
        compute_dtype=torch.float32,
    )
    batch = collator([{key: value[0] for key, value in columns.items()}])
    assert batch["token_weights"].shape == batch["labels"].shape == batch["kl_gamma"].shape
    assert torch.equal(batch["token_weights"].gt(0), batch["labels"].ne(-100))
    assert not torch.equal(batch["token_weights"], batch["kl_gamma"])
    args = TrainingArguments(
        output_dir=str(tmp_path / "output"), use_cpu=True, report_to=[], remove_unused_columns=False
    )
    trainer = CustomSeq2SeqTrainer(
        finetuning_args=FinetuningArguments(stage="sft", finetuning_type="full"),
        processor=None,
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer._webkd_kl_ref = reference
    loss, outputs = trainer.compute_loss(model, batch, return_outputs=True)
    # Identical initial/reference models must make the anchor vanish.
    expected = weighted_lm_loss(outputs.logits, batch["labels"], batch["token_weights"])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    assert all(p.grad is None for p in reference.parameters())
    assert torch.isfinite(loss)
