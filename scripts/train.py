"""Fine-tune GLiNER (urchade/gliner_multi-v2.1) to find Kazakh profanity spans.

Training objective: for every candidate span (1..12 words) and the single label
"profanity", GLiNER predicts a probability; the loss is focal binary
cross-entropy between that probability and 1 (span is a labelled profane word)
or 0 (any other span). Clean comments contribute only 0-targets, which is how
the model learns NOT to fire on ordinary or merely rude words.

The checkpoint with the lowest validation loss is kept (load_best_model_at_end).
The decision threshold is NOT chosen here - see scripts/evaluate.py.

Usage:  python scripts/train.py [--config configs/train.json] [--max-steps N]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


class FreeMpsCache:
    """Apple-GPU (MPS) caches buffers for every new tensor shape; with batches of
    different lengths that cache grows until the machine swaps. Emptying it after
    each step keeps memory flat. No effect on CUDA/CPU."""

    def __init__(self):
        from transformers import TrainerCallback

        class _CB(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                    if state.global_step % 25 == 0:
                        print(f"[mem] step {state.global_step} mps_driver_gb="
                              f"{torch.mps.driver_allocated_memory() / 2**30:.1f}", flush=True)

            on_evaluate = on_step_end

        self.callback = _CB()


def to_gliner_example(row: dict, splitter, label: str) -> dict:
    """Character spans -> word-index spans in GLiNER's own tokenisation.

    'Қо*ақбас екен' is split into ['Қо', '*', 'ақбас', 'екен']; the span covering
    characters 0..8 becomes words 0..2 (inclusive)."""
    tokens, starts, ends = [], [], []
    for tok, s, e in splitter(row["text"]):
        tokens.append(tok)
        starts.append(s)
        ends.append(e)
    ner = []
    for cs, ce in row["spans"]:
        idx = [i for i, (s, e) in enumerate(zip(starts, ends)) if s < ce and e > cs]
        if idx:
            ner.append([idx[0], idx[-1], label])
    # ner_labels forces the "profanity" prompt on EVERY example, so clean
    # comments are real negatives instead of being skipped.
    return {"tokenized_text": tokens, "ner": ner, "ner_labels": [label]}


def make_trainer(model, train, val, training_args):
    """Same as GLiNER.train_model(), plus the MPS memory callback."""
    from gliner.training import Trainer

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=model._create_data_collator(),
        processing_class=model.data_processor.transformer_tokenizer,
        callbacks=[FreeMpsCache().callback],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "train.json"))
    ap.add_argument("--max-steps", type=int, default=-1, help="debug: stop after N steps")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    set_seed(cfg["seed"])

    from gliner import GLiNER

    model = GLiNER.from_pretrained(cfg["base_model"])
    model.config.max_len = cfg["max_len"]
    if cfg.get("freeze_word_embeddings", False):
        # The word-embedding matrix is 250k x 768 = 192M of the 289M parameters.
        # Freezing it saves ~2.2 GB of gradient + Adam state (needed to fit an
        # Apple-silicon GPU) and limits overfitting on ~8k examples; the 12
        # transformer layers and the GLiNER span/label heads still train.
        # The embedding LayerNorm is frozen too: on MPS its backward returns NaN
        # when the embedding output does not require grad (found while debugging).
        for p in model.model.token_rep_layer.bert_layer.model.embeddings.parameters():
            p.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train / 1e6:.1f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    splitter = model.data_processor.words_splitter

    data = ROOT / "data" / "processed"
    train = [to_gliner_example(r, splitter, cfg["label"]) for r in load_jsonl(data / "train.jsonl")]
    val = [to_gliner_example(r, splitter, cfg["label"]) for r in load_jsonl(data / "val.jsonl")]
    print(f"train {len(train)} (with spans {sum(bool(x['ner']) for x in train)}), val {len(val)}")

    steps_per_epoch = -(-len(train) // (cfg["per_device_train_batch_size"] * cfg.get("gradient_accumulation_steps", 1)))
    eval_steps = max(50, steps_per_epoch // 2)
    out_dir = ROOT / cfg["output_dir"]
    training_args = model.create_training_args(
        output_dir=str(out_dir),
        learning_rate=cfg["learning_rate"],
        others_lr=cfg["others_lr"],
        weight_decay=cfg["weight_decay"],
        others_weight_decay=cfg["others_weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        focal_loss_alpha=cfg["focal_loss_alpha"],
        focal_loss_gamma=cfg["focal_loss_gamma"],
        max_grad_norm=cfg["max_grad_norm"],
        max_steps=args.max_steps,
        num_train_epochs=cfg["num_train_epochs"],
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=1,          # best + last are kept; disk is tight
        save_only_model=True,        # weights only (no optimizer state): ~1.2 GB per checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        dataloader_num_workers=0,
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        report_to="none",
    )
    trainer = make_trainer(model, train, val, training_args)
    trainer.train()

    final = ROOT / cfg["final_dir"]
    trainer.model.save_pretrained(str(final))
    (final / "training_config.json").write_text(json.dumps(cfg, indent=2))
    history = [h for h in trainer.state.log_history]
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "training_log.json").write_text(json.dumps(history, indent=2))
    print("best checkpoint:", trainer.state.best_model_checkpoint, "best eval_loss:", trainer.state.best_metric)
    shutil.rmtree(out_dir, ignore_errors=True)   # checkpoints are 1+ GB each
    print("saved model to", final)


if __name__ == "__main__":
    main()
