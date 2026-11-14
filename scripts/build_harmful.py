"""Build data/harmful_intent.jsonl from public sets. Run once; needs network.

Sources (all ungated):
  lmsys/toxic-chat            real user prompts, human toxicity + jailbreak labels   -> train/val/test, tags jailbreak when flagged
  JailbreakBench/JBB-Behaviors harmful + benign behavior pairs                      -> test, tag 'jbb'
  tatsu-lab/alpaca            benign instructions, precision tail                    -> all splits, negatives
Harmful positives are intentionally NOT from a single taxonomy so 'source' slices reveal probe overfit.
"""
import random, sys
sys.path.insert(0, ".")
from datasets import load_dataset
from probelab.schema import Example, write_jsonl

random.seed(0)
C = "harmful_intent"
exs = []

def split(): return random.choices(["train", "val", "test"], [0.7, 0.15, 0.15])[0]

tc = load_dataset("lmsys/toxic-chat", "toxicchat0124")
for sp in ["train", "test"]:
    for i, r in enumerate(tc[sp]):
        tags = ["toxicchat"] + (["jailbreak"] if r["jailbreaking"] else [])
        exs.append(Example(id=f"tc-{sp}-{i}", text=r["user_input"], labels={C: int(r["toxicity"])}, split=split() if sp == "train" else "test", source="toxic-chat", tags=tags))

jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
for name, y in [("harmful", 1), ("benign", 0)]:
    for i, r in enumerate(jbb[name]):
        exs.append(Example(id=f"jbb-{name}-{i}", text=r["Goal"], labels={C: y}, split="test", source="jbb", tags=["jbb", f"cat:{r['Category']}"]))

al = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=0).select(range(4000))
for i, r in enumerate(al):
    t = r["instruction"] + ("\n" + r["input"] if r["input"] else "")
    exs.append(Example(id=f"alpaca-{i}", text=t, labels={C: 0}, split=split(), source="alpaca", tags=["benign-pool"]))

write_jsonl("data/harmful_intent.jsonl", exs)
from collections import Counter
print(len(exs), Counter((e.split, e.labels[C]) for e in exs))
