"""Evaluate PHI detection on gold synthetic notes.

Modes:
  --model M [--adapter A]   LLM (prompt-only if no adapter)
  --regex                   rule-based baseline

Main metric = span recall: a gold PHI span counts as caught if its exact text is
in the predicted set (masking replaces every occurrence of a predicted string).

Usage: python synth/evaluate.py --gold data/sft/test.gold.jsonl --model models/X --adapter adapters/Y --out results/x.json
"""
import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from build_sft import SYSTEM

REGEXES = {
    "DATE": r"\b(?:19|20)\d{2}[/-]\d{1,2}[/-]\d{1,2}\b|\b1[01]\d/\d{2}/\d{2}\b|\b\d{1,2}/\d{1,2}\b|\b[A-Z][a-z]{2} \d{2}, \d{4}\b",
    "PHONE": r"09\d{2}-\d{3}-\d{3}|\(0\d\) \d{4}-\d{4}",
    "IDNO": r"\b[A-Z][12]\d{8}\b",
    "MRN": r"\b\d{7,8}\b",
    "HOSPITAL": r"[一-鿿]{2}(?:綜合醫院|紀念醫院|醫學中心|聯合診所|醫院|診所)",
}


def regex_predict(text):
    return [{"type": t, "text": m.group(0)} for t, rx in REGEXES.items() for m in re.finditer(rx, text)]


def parse(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    raw = re.sub(r"<\|channel>.*?<channel\|>", "", raw, flags=re.S)
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return [p for p in obj.get("phi", []) if isinstance(p, dict) and isinstance(p.get("text"), str)]
    except (json.JSONDecodeError, AttributeError):
        return None


def score(golds, preds):
    per_note, by_type = [], defaultdict(lambda: [0, 0])
    tp_u = pred_u = 0
    for g, p in zip(golds, preds):
        ptexts = {x["text"] for x in (p or []) if x["text"].strip()}
        caught = [s["text"] in ptexts for s in g["phi"]]
        for s, c in zip(g["phi"], caught):
            by_type[s["type"]][0] += c
            by_type[s["type"]][1] += 1
        gtexts = {s["text"] for s in g["phi"]}
        tp_u += len(ptexts & gtexts)
        pred_u += len(ptexts)
        per_note.append((sum(caught), len(caught)))
    rec = sum(a for a, _ in per_note) / sum(b for _, b in per_note)
    rng = random.Random(0)
    boots = []
    for _ in range(2000):
        sample = [per_note[rng.randrange(len(per_note))] for _ in per_note]
        boots.append(sum(a for a, _ in sample) / max(1, sum(b for _, b in sample)))
    boots.sort()
    prec = tp_u / pred_u if pred_u else 0.0
    return {
        "n_notes": len(golds),
        "span_recall": round(rec, 4),
        "span_recall_95ci": [round(boots[49], 4), round(boots[1949], 4)],
        "precision_unique": round(prec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "clean_note_rate": round(sum(a == b for a, b in per_note) / len(per_note), 4),
        "valid_json_rate": round(sum(p is not None for p in preds) / len(preds), 4),
        "recall_by_type": {t: round(a / b, 4) for t, (a, b) in sorted(by_type.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--regex", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    golds = [json.loads(l) for l in open(args.gold, encoding="utf-8")][: args.limit]
    t0 = time.time()
    raws = []
    if args.regex:
        preds = [regex_predict(g["text"]) for g in golds]
    else:
        from mlx_lm import batch_generate, load
        model, tok = load(args.model, adapter_path=args.adapter)
        preds = []
        for i in range(0, len(golds), args.batch):
            chunk = golds[i:i + args.batch]
            prompts = [tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": g["text"]}],
                add_generation_prompt=True, enable_thinking=False) for g in chunk]
            res = batch_generate(model, tok, prompts, max_tokens=500)
            raws += res.texts
            preds += [parse(r) for r in res.texts]
            print(f"{len(preds)}/{len(golds)} {time.time() - t0:.0f}s", flush=True)
    result = {"mode": "regex" if args.regex else ("lora" if args.adapter else "prompt-only"),
              "model": args.model, "adapter": args.adapter, "wall_s": round(time.time() - t0), **score(golds, preds)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    with open(Path(args.out).with_suffix(".preds.jsonl"), "w", encoding="utf-8") as f:
        for g, p, r in zip(golds, preds, raws or [None] * len(preds)):
            f.write(json.dumps({"id": g["id"], "pred": p, "raw": r}, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
