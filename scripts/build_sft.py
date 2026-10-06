"""Turn filled notes into chat-format SFT splits for mlx-lm / Unsloth.

Target output is a JSON list of PHI spans (not the rewritten note): short to
generate, easy to score, and Python does the actual masking afterwards.

Usage: python synth/build_sft.py --in data/notes.jsonl --out data/sft
"""
import argparse
import json
import random
from pathlib import Path

SYSTEM = (
    "你是病歷去識別化助手。找出病歷中所有個人可識別資訊（PHI），"
    "類型限 NAME, DOCTOR, MRN, IDNO, DATE, PHONE, ADDRESS, HOSPITAL。"
    "只輸出 JSON：{\"phi\": [{\"type\": 類型, \"text\": 原文字串}]}，依出現順序，重複出現也要列出。"
)


def target(rec):
    return json.dumps({"phi": [{"type": p["type"], "text": p["text"]} for p in rec["phi"]]}, ensure_ascii=False)


def to_chat(rec):
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": rec["text"]},
        {"role": "assistant", "content": target(rec)},
    ]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="0.7,0.1,0.2")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    # De-duplicate identical note texts before splitting so test never leaks into train.
    seen, uniq = set(), []
    for r in recs:
        if r["text"] not in seen:
            seen.add(r["text"])
            uniq.append(r)
    random.Random(args.seed).shuffle(uniq)
    a, b, _ = (float(x) for x in args.split.split(","))
    n = len(uniq)
    parts = {"train": uniq[: int(n * a)], "valid": uniq[int(n * a): int(n * (a + b))], "test": uniq[int(n * (a + b)):]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in parts.items():
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(to_chat(r), ensure_ascii=False) + "\n")
        # Keep gold records (with offsets) for evaluation.
        with (out / f"{name}.gold.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print({k: len(v) for k, v in parts.items()}, f"dropped_dupes={len(recs) - n}")


if __name__ == "__main__":
    main()
