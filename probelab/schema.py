"""Data schema. One JSONL line per example.

{
  "id": "abc",
  "text": "...",                      # or "messages": [{"role":..,"content":..}]
  "labels": {"harmful_intent": 1},    # concept -> 0/1 ; missing concept = unlabeled for it
  "split": "train" | "val" | "test",
  "source": "toxic-chat",
  "tags": ["paraphrase", "multi_turn", "jailbreak", "lang:es"]   # distribution-shift slices
}
Concepts are just strings. General concepts we control (harmful_intent, deception, ...)
and custom user-aligned concepts (refund_policy_violation, ...) share the same pipeline.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Example:
    id: str
    text: str | None = None
    messages: list[dict] | None = None
    labels: dict[str, int] = field(default_factory=dict)
    split: str = "train"
    source: str = ""
    tags: list[str] = field(default_factory=list)

    def render(self, tokenizer=None) -> str:
        if self.messages is not None:
            if tokenizer is not None and getattr(tokenizer, "chat_template", None):
                return tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
            return "\n".join(f"{m['role']}: {m['content']}" for m in self.messages)
        assert self.text is not None, f"example {self.id} has neither text nor messages"
        return self.text


def read_jsonl(path: str | Path) -> list[Example]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(Example(**d))
    return out


def write_jsonl(path: str | Path, examples: Iterator[Example] | list[Example]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.__dict__, ensure_ascii=False) + "\n")
