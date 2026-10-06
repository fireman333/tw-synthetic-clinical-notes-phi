"""Stage 1: generate note skeletons (placeholders instead of identifiers).

Usage: python synth/gen_skeletons.py --model models/X --n 50 --out data/skeletons.jsonl
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import mlx.core as mx
import opencc
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

from spec import NOTE_TYPES, PHI_TYPES, SKELETON_PROMPT, SPECIALTIES, STYLES, TOPICS, TW_TERMS

# s2tw = character-level only; s2twp rewrote already-traditional phrases (局部 -> 區域性).
S2TW = opencc.OpenCC("s2tw")
SYS = "你是台灣醫學中心的住院醫師。直接輸出病歷本文，不要輸出思考過程。"
PH_RE = re.compile(r"\{([A-Z]+)\}")
# Things that look like leaked identifiers outside placeholders.
LEAK_RES = {
    "phone": re.compile(r"09\d{2}[-\s]?\d{3}[-\s]?\d{3}|\(0\d\)\s?\d{3,4}-?\d{4}"),
    "idno": re.compile(r"\b[A-Z][12]\d{8}\b"),
    "date": re.compile(r"\b(19|20)\d{2}[/-]\d{1,2}[/-]\d{1,2}\b|\b1[01]\d/\d{1,2}/\d{1,2}\b"),
    "long_number": re.compile(r"\b\d{7,}\b"),
    "hospital": re.compile(r"[一-鿿]{2,6}(醫院|診所|醫學中心)"),
}


def build_job(rng):
    note_type = rng.choice(list(NOTE_TYPES))
    specialty = rng.choice(SPECIALTIES)
    if specialty == "小兒科":
        age = rng.choice([0, 1, 2, 3, 5, 7, 10, 14])
    elif specialty == "婦產科":
        age = rng.randint(18, 45)
    else:
        age = rng.randint(20, 92)
    sex = "女性" if specialty == "婦產科" else rng.choice(["男性", "女性"])
    topic = rng.choice(TOPICS[specialty])
    # Sex-specific topics.
    if re.search(r"乳房|子宮|卵巢|妊娠|產|子癇", topic):
        sex = "女性"
    elif re.search(r"攝護腺|睪丸", topic):
        sex = "男性"
    if topic == "新生兒黃疸":
        age = 0
    # Age windows for age-typical diseases (QA found e.g. 14-year-old bronchiolitis).
    for pat, lo, hi in [(r"細支氣管炎", 0, 2), (r"熱性痙攣", 1, 5), (r"川崎病", 1, 5), (r"手足口病", 1, 6),
                        (r"COPD|間質性肺病|多發性骨髓瘤|糖尿病視網膜", 50, 88)]:
        if re.search(pat, topic):
            age = rng.randint(lo, hi)
    if re.search(r"攝護腺肥大|主動脈瓣狹窄|白內障|股骨頸骨折|全膝置換|慢性硬腦膜下", topic):
        age = max(age, rng.randint(55, 90))
    must = rng.sample(["{MRN}", "{DOCTOR}", "{HOSPITAL}", "{PHONE}", "{ADDRESS}", "{IDNO}"], k=2)
    return {
        "specialty": specialty,
        "topic": topic,
        "age": age,
        "sex": sex,
        "note_type": note_type,
        "style": rng.choice(list(STYLES)),
        "length": rng.choice([150, 250, 350]),
        "min_ph": rng.choice([3, 4, 5]),
        "must": must,
    }


def make_prompt(job):
    return SKELETON_PROMPT.format(
        note_type=job["note_type"],
        note_type_desc=NOTE_TYPES[job["note_type"]],
        specialty=job["specialty"],
        topic=job["topic"],
        age=job["age"],
        sex=job["sex"],
        style_desc=STYLES[job["style"]],
        length=job["length"],
        min_ph=job["min_ph"],
        must="、".join(job["must"]),
    )


def clean(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    # Gemma 4 sometimes emits a thought channel even with thinking disabled.
    text = re.sub(r"<\|channel>.*?<channel\|>", "", text, flags=re.S)
    text = text.replace("```", "").strip()
    # Normalise placeholder variants such as {{NAME}} or 【NAME】.
    text = re.sub(r"\{\{([A-Z]+)\}\}", r"{\1}", text)
    text = S2TW.convert(text)
    for cn, tw in TW_TERMS.items():
        text = text.replace(cn, tw)
    return text


def check(text):
    """Return list of problems; empty list = accepted."""
    problems = []
    if re.search(r"Thinking Process|<\|channel>|\*\*Analyze", text):
        problems.append("thinking text leaked")
    found = PH_RE.findall(text)
    unknown = sorted({p for p in found if p not in PHI_TYPES})
    if unknown:
        problems.append(f"unknown placeholders {unknown}")
    if "NAME" not in found or "DATE" not in found:
        problems.append("missing NAME or DATE")
    if len(found) < 3:
        problems.append("fewer than 3 placeholders")
    stripped = PH_RE.sub("", text)
    for kind, rx in LEAK_RES.items():
        if rx.search(stripped):
            problems.append(f"possible leaked {kind}: {rx.search(stripped).group(0)}")
    if re.search(r"^\s*#|\*\*|\\text|\[Your|以下是", text, flags=re.M):
        problems.append("markdown / template residue")
    if re.search(r"\$[^$\n]+\$|\\\(|\\frac|\\mathrm", text):
        problems.append("LaTeX")
    if re.search(r"[\u0900-\u097F\u0E00-\u0E7F\u0400-\u04FF\u3040-\u30FF\uAC00-\uD7AF\u0600-\u06FF]", text):
        problems.append("foreign script")
    if re.search(r"\}\s*\{[A-Z]+\}\s*\{", text):
        problems.append("stacked placeholders")
    if len(text) < 80:
        problems.append("too short")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=900)
    args = ap.parse_args()

    out = Path(args.out)
    # Resumable: keep already-accepted skeletons and continue with a shifted seed.
    done = sum(1 for _ in out.open(encoding="utf-8")) if out.exists() else 0
    rng = random.Random(args.seed + done)
    model, tok = load(args.model)
    sampler = make_sampler(temp=0.9, top_p=0.95)
    out.parent.mkdir(parents=True, exist_ok=True)

    accepted, rejected = done, 0
    oom_streak = 0
    t0 = time.time()
    with out.open("a", encoding="utf-8") as fo, out.with_suffix(".rejected.jsonl").open("a", encoding="utf-8") as fr:
        while accepted < args.n:
            jobs = [build_job(rng) for _ in range(args.batch)]
            prompts = [
                tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": make_prompt(j)}],
                                        add_generation_prompt=True, enable_thinking=False)
                for j in jobs
            ]
            try:
                res = batch_generate(model, tok, prompts, max_tokens=args.max_tokens, sampler=sampler)
                oom_streak = 0
            except RuntimeError as e:
                # Another process can grab GPU memory; back off instead of dying.
                if "Insufficient Memory" not in str(e) or oom_streak >= 5:
                    raise
                oom_streak += 1
                args.batch = max(1, args.batch // 2)
                mx.clear_cache()
                print(f"OOM -> sleep 30s, batch={args.batch}, streak={oom_streak}", flush=True)
                time.sleep(30)
                continue
            for job, raw in zip(jobs, res.texts):
                text = clean(raw)
                problems = check(text)
                rec = {**job, "skeleton": text, "generator_model": Path(args.model).name}
                if problems:
                    rejected += 1
                    fr.write(json.dumps({**rec, "problems": problems}, ensure_ascii=False) + "\n")
                elif accepted < args.n:
                    accepted += 1
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
            print(f"accepted {accepted}/{args.n}  rejected {rejected}  {time.time() - t0:.0f}s", flush=True)
    print(f"DONE accept_rate={(accepted - done) / max(1, accepted - done + rejected):.2f} wall={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
