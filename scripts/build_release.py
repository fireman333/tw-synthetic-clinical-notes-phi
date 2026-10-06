"""Assemble the public HF release from QA-approved skeletons.

skeletons.jsonl (line index = skeleton id) + data/qa/*.verdict.jsonl (keep/drop)
 -> fill PHI (one fill per skeleton) -> split by skeleton -> HF jsonl + SFT chat jsonl.

Usage: python synth/build_release.py --n 1000 --out data/release/hf
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from build_sft import to_chat
from fill_phi import Faker_TW, fill

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("id", "specialty", "topic", "note_type", "style", "text", "phi", "deid_text")


def load_verdicts():
    """Map skeleton index -> verdict. QA batch files carry 'offset' = first skeleton index."""
    verdicts = {}
    for vf in sorted((ROOT / "data/qa").glob("batch_*.verdict.jsonl")):
        meta = json.loads(vf.with_name(vf.name.replace(".verdict.jsonl", ".meta.json")).read_text())
        for line in vf.open(encoding="utf-8"):
            v = json.loads(line)
            verdicts[meta["offset"] + v["idx"]] = v
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(ROOT / "data/release/hf"))
    args = ap.parse_args()

    skels = [json.loads(l) for l in (ROOT / "data/release/skeletons.jsonl").open(encoding="utf-8")]
    verdicts = load_verdicts()
    unreviewed = [i for i in range(len(skels)) if i not in verdicts]
    kept = [i for i in range(len(skels)) if verdicts.get(i, {}).get("verdict") == "keep"]
    # De-duplicate identical skeletons.
    seen, uniq = set(), []
    for i in kept:
        if skels[i]["skeleton"] not in seen:
            seen.add(skels[i]["skeleton"])
            uniq.append(i)
    print(f"skeletons={len(skels)} reviewed={len(verdicts)} unreviewed={len(unreviewed)} keep={len(kept)} uniq={len(uniq)}")
    if len(uniq) < args.n:
        raise SystemExit(f"only {len(uniq)} approved skeletons, need {args.n}")

    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(uniq, args.n))
    rng.shuffle(chosen)
    n_tr, n_va = int(args.n * 0.7), int(args.n * 0.1)
    splits = {"train": chosen[:n_tr], "validation": chosen[n_tr:n_tr + n_va], "test": chosen[n_tr + n_va:]}

    fk = Faker_TW(args.seed)
    out = Path(args.out)
    (out / "sft").mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, idxs in splits.items():
        recs = []
        for i in idxs:
            s = skels[i]
            text, spans, deid = fill(s["skeleton"], fk)
            recs.append({"id": f"twsyn-{i:04d}", "specialty": s["specialty"], "topic": s.get("topic", ""),
                         "note_type": s["note_type"], "style": s["style"], "text": text, "phi": spans,
                         "deid_text": deid})
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                assert tuple(r) == FIELDS
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with (out / "sft" / f"{'valid' if name == 'validation' else name}.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(to_chat(r), ensure_ascii=False) + "\n")
        stats[name] = {
            "n": len(recs),
            "phi_spans": sum(len(r["phi"]) for r in recs),
            "avg_chars": sum(len(r["text"]) for r in recs) / len(recs),
            "phi_types": dict(Counter(p["type"] for r in recs for p in r["phi"])),
        }
    stats["specialty"] = dict(Counter(skels[i]["specialty"] for i in chosen))
    stats["note_type"] = dict(Counter(skels[i]["note_type"] for i in chosen))
    stats["style"] = dict(Counter(skels[i]["style"] for i in chosen))
    stats["qa"] = {"reviewed": len(verdicts), "kept": len(kept)}
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    write_card(out, stats)
    print(json.dumps({k: v for k, v in stats.items() if k in ("train", "validation", "test", "qa")}, ensure_ascii=False))


def write_card(out, stats):
    """Fill README.template.md placeholders from stats."""
    splits = [stats[k] for k in ("train", "validation", "test")]
    total = sum(x["n"] for x in splits)
    phi = Counter()
    for x in splits:
        phi.update(x["phi_types"])
    vals = {
        "N_TOTAL": f"{total:,}", "N_TRAIN": splits[0]["n"], "N_VALID": splits[1]["n"], "N_TEST": splits[2]["n"],
        "N_REVIEWED": f"{stats['qa']['reviewed']:,}", "N_KEPT": f"{stats['qa']['kept']:,}",
        "AVG_PHI": f"{sum(phi.values()) / total:.1f}",
        "AVG_CHARS": f"{sum(x['avg_chars'] * x['n'] for x in splits) / total:.1f}",
        **{f"PHI_{t}": f"{phi.get(t, 0):,}" for t in ("NAME", "DOCTOR", "MRN", "IDNO", "DATE", "PHONE", "ADDRESS", "HOSPITAL")},
    }
    card = (ROOT / "data/release/README.template.md").read_text(encoding="utf-8")
    for k, v in vals.items():
        card = card.replace("{{" + k + "}}", str(v))
    assert "{{" not in card, "unfilled card placeholder"
    (out / "README.md").write_text(card, encoding="utf-8")


if __name__ == "__main__":
    main()
