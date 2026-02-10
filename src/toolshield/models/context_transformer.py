"""Context-augmented transformer classifier for prompt injection detection.

Extends the base transformer classifier to include contextual information
(role, tool name, schema, description) alongside the user prompt.

Supports two truncation strategies:
- "naive": Concatenate all fields into one string, let the tokenizer
  right-truncate to max_length. This silently clips the prompt (which
  appears last) when tool schemas are large — a subtle failure mode
  in enterprise deployments where schemas can consume hundreds of tokens.
- "keep_prompt": Tokenize context and prompt separately, guaranteeing
  a minimum token budget for the prompt tail. Context is truncated to
  fit the remainder. This preserves the attacker-controlled signal and
  improves detection under fixed token budgets.

Input format (naive):
  [CLS] ROLE: {roles} [SEP] TOOL: {name} [SEP] SCHEMA: {schema}
  [SEP] DESC: {desc} [SEP] PROMPT: {prompt} [SEP]

Input format (keep_prompt):
  <s> context_tokens </s></s> prompt_tokens </s>
  where context = "ROLE: ... [SEP] TOOL: ... [SEP] SCHEMA: ... [SEP] DESC: ..."
  and   prompt  = "PROMPT: {prompt}"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from toolshield.data.schema import DatasetRecord
from toolshield.models.transformer import TransformerClassifier

logger = logging.getLogger(__name__)


# ── Token retention statistics ───────────────────────────────────────────────

@dataclass
class TruncationStats:
    """Token retention statistics for a set of records.

    Provides empirical evidence of how truncation strategies affect
    prompt vs. context token survival under fixed max_length budgets.
    """

    n_samples: int
    prompt_tokens_total: list[int]       # raw prompt token counts per sample
    prompt_tokens_retained: list[int]    # prompt tokens after truncation
    context_tokens_total: list[int]      # raw context token counts per sample
    context_tokens_retained: list[int]   # context tokens after truncation
    n_prompt_truncated: int              # samples where prompt was clipped

    @property
    def prompt_retention_ratio(self) -> list[float]:
        """Per-sample ratio of retained / total prompt tokens."""
        return [
            r / t if t > 0 else 1.0
            for r, t in zip(self.prompt_tokens_retained, self.prompt_tokens_total)
        ]

    def summary(self) -> dict[str, float]:
        """Compute aggregate statistics for reporting."""
        pr = np.array(self.prompt_retention_ratio)
        pt = np.array(self.prompt_tokens_retained)
        ct = np.array(self.context_tokens_retained)
        return {
            "n_samples": self.n_samples,
            "n_prompt_truncated": self.n_prompt_truncated,
            "pct_prompt_truncated": self.n_prompt_truncated / max(self.n_samples, 1),
            "prompt_retained_mean": float(np.mean(pt)),
            "prompt_retained_p50": float(np.median(pt)),
            "prompt_retained_p95": float(np.percentile(pt, 95)),
            "context_retained_mean": float(np.mean(ct)),
            "context_retained_p50": float(np.median(ct)),
            "context_retained_p95": float(np.percentile(ct, 95)),
            "prompt_retention_ratio_mean": float(np.mean(pr)),
            "prompt_retention_ratio_p50": float(np.median(pr)),
            "prompt_retention_ratio_p05": float(np.percentile(pr, 5)),
        }


# ── Prompt-preserving dataset ───────────────────────────────────────────────

class PromptPreservingDataset(TorchDataset):
    """PyTorch Dataset with prompt-preserving truncation.

    Tokenizes context and prompt independently, then assembles them
    into a single sequence that fits max_length while guaranteeing
    the prompt retains at least ``prompt_min_tokens`` tokens.

    Token layout (RoBERTa):
        <s> context_tokens </s></s> prompt_tokens </s>
        |------- 4 special tokens overhead ---------|

    Budget allocation:
        content_budget = max_length - 4
        prompt gets min(prompt_min_tokens, content_budget) reserved
        context gets content_budget - prompt_reserved
        if context is shorter, prompt gets the surplus
    """

    # <s>, </s> after context, </s> before prompt, </s> after prompt
    SPECIAL_TOKENS_COUNT = 4

    def __init__(
        self,
        contexts: list[str],
        prompts: list[str],
        labels: list[int] | None,
        tokenizer: Any,
        max_length: int = 256,
        prompt_min_tokens: int = 128,
        prompt_side: str = "tail",
    ) -> None:
        assert len(contexts) == len(prompts), "contexts and prompts must align"
        if labels is not None:
            assert len(labels) == len(prompts), "labels must align with prompts"

        self.contexts = contexts
        self.prompts = prompts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_min_tokens = prompt_min_tokens
        self.prompt_side = prompt_side

        # Cache special token IDs
        self.bos_id = tokenizer.bos_token_id   # <s>
        self.eos_id = tokenizer.eos_token_id   # </s>
        self.pad_id = tokenizer.pad_token_id   # <pad>

    def __len__(self) -> int:
        return len(self.prompts)

    def _allocate_and_build(
        self, context: str, prompt: str
    ) -> tuple[list[int], list[int], int, int, int, int]:
        """Tokenize context + prompt with prompt-preserving allocation.

        Two-pass allocation:
          1. Reserve prompt_min_tokens for prompt; context gets the rest.
          2. If prompt used less than its reservation, expand context.

        Returns:
            (input_ids, attention_mask,
             prompt_total, prompt_retained,
             context_total, context_retained)
        """
        content_budget = self.max_length - self.SPECIAL_TOKENS_COUNT

        # Tokenize both without special tokens
        prompt_ids_full = self.tokenizer.encode(prompt, add_special_tokens=False)
        context_ids_full = self.tokenizer.encode(context, add_special_tokens=False)

        prompt_total = len(prompt_ids_full)
        context_total = len(context_ids_full)

        # Pass 1: reserve tokens for prompt; context gets the remainder
        prompt_reserved = min(self.prompt_min_tokens, content_budget)
        context_max = content_budget - prompt_reserved

        context_ids = context_ids_full[:context_max]

        # Prompt gets whatever is left after context
        prompt_budget = content_budget - len(context_ids)
        if len(prompt_ids_full) > prompt_budget:
            if self.prompt_side == "tail":
                prompt_ids = prompt_ids_full[-prompt_budget:]
            else:
                prompt_ids = prompt_ids_full[:prompt_budget]
        else:
            prompt_ids = list(prompt_ids_full)

        # Pass 2: rebalance — if prompt used less than its budget, expand context
        slack = content_budget - len(context_ids) - len(prompt_ids)
        if slack > 0 and len(context_ids) < context_total:
            extra = min(slack, context_total - len(context_ids))
            context_ids = context_ids_full[: len(context_ids) + extra]

        prompt_retained = len(prompt_ids)
        context_retained = len(context_ids)

        # Assemble: <s> context </s></s> prompt </s>
        input_ids = (
            [self.bos_id]
            + context_ids
            + [self.eos_id, self.eos_id]
            + prompt_ids
            + [self.eos_id]
        )

        # Pad to max_length
        attention_mask = [1] * len(input_ids)
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return (
            input_ids, attention_mask,
            prompt_total, prompt_retained,
            context_total, context_retained,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        context = self.contexts[idx]
        prompt = self.prompts[idx]

        input_ids, attention_mask, _, _, _, _ = self._allocate_and_build(
            context, prompt
        )

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


# ── Context transformer classifier ──────────────────────────────────────────

class ContextTransformerClassifier(TransformerClassifier):
    """Context-augmented transformer with configurable truncation strategy.

    Extends TransformerClassifier to include role, tool, schema, and
    description context in the input representation.

    Truncation strategies:
        "naive"       — concatenate everything, right-truncate (baseline).
        "keep_prompt" — reserve prompt_min_tokens for the prompt tail,
                        truncate context to fit the remainder.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)

        # Context field options
        self.include_schema = self.config.get("include_schema", True)
        self.include_description = self.config.get("include_description", True)
        self.max_schema_length = self.config.get("max_schema_length", 200)

        # Truncation strategy
        self.truncate_strategy = self.config.get("truncate_strategy", "naive")
        self.prompt_min_tokens = self.config.get("prompt_min_tokens", 128)
        self.prompt_side = self.config.get("prompt_side", "tail")

        if self.truncate_strategy not in ("naive", "keep_prompt"):
            raise ValueError(
                f"Unknown truncate_strategy: {self.truncate_strategy!r}. "
                f"Must be 'naive' or 'keep_prompt'."
            )

    # ── Formatting helpers ───────────────────────────────────────────────

    def _format_context(self, record: DatasetRecord) -> str:
        """Format context fields WITHOUT the prompt.

        Returns:
            String like "ROLE: user [SEP] TOOL: foo [SEP] SCHEMA: {...} [SEP] DESC: ..."
        """
        parts = []

        role_str = ", ".join(record.role_sequence)
        parts.append(f"ROLE: {role_str}")

        parts.append(f"TOOL: {record.tool_name}")

        if self.include_schema:
            schema_str = json.dumps(record.tool_schema, separators=(",", ":"))
            if len(schema_str) > self.max_schema_length:
                schema_str = schema_str[: self.max_schema_length] + "..."
            parts.append(f"SCHEMA: {schema_str}")

        if self.include_description:
            parts.append(f"DESC: {record.tool_description}")

        return " [SEP] ".join(parts)

    @staticmethod
    def _format_prompt(record: DatasetRecord) -> str:
        """Format the prompt field with its prefix marker."""
        return f"PROMPT: {record.prompt}"

    def _format_record(self, record: DatasetRecord) -> str:
        """Format a full record (context + prompt) as a single string.

        Used by the 'naive' truncation strategy.
        """
        return self._format_context(record) + " [SEP] " + self._format_prompt(record)

    def _get_texts(self, records: list[DatasetRecord]) -> list[str]:
        """Extract context-augmented text inputs (naive strategy)."""
        return [self._format_record(r) for r in records]

    # ── Dataset / encoding overrides ─────────────────────────────────────

    def _make_dataset(
        self,
        records: list[DatasetRecord],
        labels: list[int] | None,
    ) -> TorchDataset:
        """Create dataset with the configured truncation strategy."""
        if self.truncate_strategy == "keep_prompt":
            contexts = [self._format_context(r) for r in records]
            prompts = [self._format_prompt(r) for r in records]
            return PromptPreservingDataset(
                contexts=contexts,
                prompts=prompts,
                labels=labels,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                prompt_min_tokens=self.prompt_min_tokens,
                prompt_side=self.prompt_side,
            )
        # Naive: fall through to parent (concatenated string + right-truncation)
        return super()._make_dataset(records, labels)

    def _prepare_batch_encodings(
        self,
        records_batch: list[DatasetRecord],
    ) -> dict[str, torch.Tensor]:
        """Prepare tokenized encodings for an inference batch."""
        if self.truncate_strategy == "keep_prompt":
            return self._prompt_preserving_encode(records_batch)
        return super()._prepare_batch_encodings(records_batch)

    def _prompt_preserving_encode(
        self,
        records: list[DatasetRecord],
    ) -> dict[str, torch.Tensor]:
        """Encode a batch with prompt-preserving truncation for inference.

        Uses the same allocation logic as PromptPreservingDataset but
        returns batched tensors suitable for direct model forward pass.
        """
        content_budget = self.max_length - PromptPreservingDataset.SPECIAL_TOKENS_COUNT

        prompt_reserved = min(self.prompt_min_tokens, content_budget)
        context_max = content_budget - prompt_reserved

        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id

        all_input_ids = []
        all_attention_masks = []

        for record in records:
            context = self._format_context(record)
            prompt = self._format_prompt(record)

            prompt_ids_full = self.tokenizer.encode(prompt, add_special_tokens=False)
            context_ids_full = self.tokenizer.encode(context, add_special_tokens=False)

            # Pass 1: truncate context, give remaining to prompt
            context_ids = context_ids_full[:context_max]

            prompt_budget = content_budget - len(context_ids)
            if len(prompt_ids_full) > prompt_budget:
                if self.prompt_side == "tail":
                    prompt_ids = prompt_ids_full[-prompt_budget:]
                else:
                    prompt_ids = prompt_ids_full[:prompt_budget]
            else:
                prompt_ids = list(prompt_ids_full)

            # Pass 2: rebalance — expand context if prompt has slack
            slack = content_budget - len(context_ids) - len(prompt_ids)
            if slack > 0 and len(context_ids) < len(context_ids_full):
                extra = min(slack, len(context_ids_full) - len(context_ids))
                context_ids = context_ids_full[: len(context_ids) + extra]

            # Assemble
            input_ids = [bos] + context_ids + [eos, eos] + prompt_ids + [eos]
            attention_mask = [1] * len(input_ids)

            pad_len = self.max_length - len(input_ids)
            if pad_len > 0:
                input_ids += [pad] * pad_len
                attention_mask += [0] * pad_len

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)

        return {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_attention_masks, dtype=torch.long),
        }

    # ── Truncation analysis ──────────────────────────────────────────────

    def compute_truncation_stats(
        self,
        records: list[DatasetRecord],
    ) -> TruncationStats:
        """Compute token retention statistics for a set of records.

        Useful for empirical evidence in ablation studies — shows how
        much prompt signal is preserved vs. lost under each strategy.

        Args:
            records: Dataset records to analyze.

        Returns:
            TruncationStats with per-sample token counts.

        Raises:
            RuntimeError: If tokenizer is not initialized.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be initialized (train or load first)")

        content_budget = self.max_length - PromptPreservingDataset.SPECIAL_TOKENS_COUNT

        prompt_total_list: list[int] = []
        prompt_retained_list: list[int] = []
        context_total_list: list[int] = []
        context_retained_list: list[int] = []
        n_prompt_truncated = 0

        if self.truncate_strategy == "keep_prompt":
            prompt_reserved = min(self.prompt_min_tokens, content_budget)
            context_max = content_budget - prompt_reserved

            for record in records:
                ctx_str = self._format_context(record)
                pmt_str = self._format_prompt(record)

                p_ids = self.tokenizer.encode(pmt_str, add_special_tokens=False)
                c_ids = self.tokenizer.encode(ctx_str, add_special_tokens=False)

                prompt_total_list.append(len(p_ids))
                context_total_list.append(len(c_ids))

                # Pass 1: truncate context, give remaining to prompt
                c_ret = min(len(c_ids), context_max)
                remaining = content_budget - c_ret
                p_ret = min(len(p_ids), remaining)

                # Pass 2: rebalance slack to context
                slack = content_budget - c_ret - p_ret
                if slack > 0 and c_ret < len(c_ids):
                    extra = min(slack, len(c_ids) - c_ret)
                    c_ret += extra

                context_retained_list.append(c_ret)
                prompt_retained_list.append(p_ret)

                if p_ret < len(p_ids):
                    n_prompt_truncated += 1
        else:
            # Naive: simulate right-truncation of concatenated string
            for record in records:
                full_text = self._format_record(record)
                pmt_str = self._format_prompt(record)

                full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
                p_ids = self.tokenizer.encode(pmt_str, add_special_tokens=False)

                prompt_total_list.append(len(p_ids))
                # Context = everything before prompt tokens
                c_total = len(full_ids) - len(p_ids)
                context_total_list.append(max(c_total, 0))

                # After right-truncation to content_budget
                retained = min(len(full_ids), content_budget)
                # Prompt tokens that survive = total retained - context tokens kept
                c_ret = min(c_total, retained)
                p_ret = max(retained - c_ret, 0)

                context_retained_list.append(c_ret)
                prompt_retained_list.append(p_ret)

                if p_ret < len(p_ids):
                    n_prompt_truncated += 1

        return TruncationStats(
            n_samples=len(records),
            prompt_tokens_total=prompt_total_list,
            prompt_tokens_retained=prompt_retained_list,
            context_tokens_total=context_total_list,
            context_tokens_retained=context_retained_list,
            n_prompt_truncated=n_prompt_truncated,
        )

    # ── Serialization ────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        if not self._is_trained or self.model is None or self.tokenizer is None:
            raise RuntimeError("Model must be trained before saving")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config_to_save = {
            "model_type": "context_transformer",
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            # Context options
            "include_schema": self.include_schema,
            "include_description": self.include_description,
            "max_schema_length": self.max_schema_length,
            # Truncation strategy
            "truncate_strategy": self.truncate_strategy,
            "prompt_min_tokens": self.prompt_min_tokens,
            "prompt_side": self.prompt_side,
        }

        with (path / "config.json").open("w") as f:
            json.dump(config_to_save, f, indent=2)

        self.model.save_pretrained(path / "model")
        self.tokenizer.save_pretrained(path / "tokenizer")

    @classmethod
    def load(cls, path: str | Path) -> "ContextTransformerClassifier":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(path)

        with (path / "config.json").open("r") as f:
            config = json.load(f)

        config.pop("model_type", None)

        instance = cls(config=config)

        instance.tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
        instance.model = AutoModelForSequenceClassification.from_pretrained(
            path / "model"
        )

        instance._is_trained = True
        return instance
