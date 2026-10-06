"""Pre-publication privacy & integrity gate for the HF release.

Fails (exit 1) on anything that could be a real identifier outside the
labelled fake PHI spans, or on label/offset inconsistencies.

Usage: python synth/privacy_check.py data/release/hf
"""
import json
import re
import sys
from pathlib import Path

import opencc

from fill_phi import CELEBRITY_NAMES, tw_id_checksum_ok

# Real Taiwanese hospitals / systems that must never appear (non-exhaustive, extend as needed).
REAL_HOSPITALS = [
    "台大", "臺大", "長庚", "馬偕", "榮總", "國泰", "新光", "亞東", "奇美", "彰基", "彰化基督教", "慈濟",
    "中國醫藥", "中國附醫", "成大", "高醫", "義大", "萬芳", "北醫", "雙和", "三總", "三軍總", "和信",
    "振興", "仁濟", "康寧", "博愛", "惠民", "慈心", "永福", "明德", "長青", "信望", "安和", "晨光", "童綜合", "光田", "秀傳", "部立", "署立", "市立聯合", "耕莘", "聖保祿", "嘉基", "門諾",
    "Mackay", "NTUH", "Chang Gung", "Veterans General", "Mayo Clinic", "Johns Hopkins", "Massachusetts General",
]
# Patterns that look like identifiers; anything matching outside a PHI span is a leak candidate.
ID_PATTERNS = {
    "tw_id": r"\b[A-Z][12]\d{8}\b",
    "mobile": r"09\d{2}[-\s]?\d{3}[-\s]?\d{3}",
    "landline": r"\(0\d\)\s?\d{3,4}-?\d{4}",
    "full_date": r"\b(?:19|20)\d{2}[/-]\d{1,2}[/-]\d{1,2}\b|\b1[01]\d/\d{1,2}/\d{1,2}\b",
    "long_number": r"\b\d{7,}\b",
    "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "street": r"[一-鿿]{1,6}(?:路|街|大道)[一-鿿\d]{0,3}段?\d+號",
}
S2TW = opencc.OpenCC("s2tw")


def masked(text, spans):
    chars = list(text)
    for s in spans:
        for k in range(s["start"], s["end"]):
            chars[k] = "█"
    return "".join(chars)


def main(root):
    root = Path(root)
    problems, n, ids, texts = [], 0, set(), set()
    for split in ("train", "validation", "test"):
        for line in (root / f"{split}.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            n += 1
            rid = r["id"]
            if rid in ids:
                problems.append((rid, "duplicate id"))
            ids.add(rid)
            if r["text"] in texts:
                problems.append((rid, "duplicate text"))
            texts.add(r["text"])
            # 1. Offsets must match exactly.
            for s in r["phi"]:
                if r["text"][s["start"]:s["end"]] != s["text"]:
                    problems.append((rid, f"offset mismatch {s}"))
                if s["type"] in ("NAME", "DOCTOR") and s["text"] in CELEBRITY_NAMES:
                    problems.append((rid, f"celebrity name: {s['text']}"))
                if s["type"] == "IDNO" and tw_id_checksum_ok(s["text"]):
                    problems.append((rid, f"IDNO passes real checksum: {s['text']}"))
            # 2. deid_text must contain no PHI string.
            for s in r["phi"]:
                # Digit-bounded so a fake date "10/12" does not match inside BP "210/120".
                if re.search(r"(?<!\d)" + re.escape(s["text"]) + r"(?!\d)", r["deid_text"]):
                    problems.append((rid, f"PHI survives in deid_text: {s['text']}"))
            # 3. Nothing identifier-like outside labelled spans.
            outside = masked(r["text"], r["phi"])
            for kind, rx in ID_PATTERNS.items():
                for m in re.finditer(rx, outside):
                    problems.append((rid, f"unlabelled {kind}: {m.group(0)}"))
            # Real hospital names: anywhere outside fake spans, or as a fake HOSPITAL value.
            # (Fake NAME / ADDRESS spans may legitimately contain 明德, 博愛路 ...)
            hosp_vals = " ".join(s["text"] for s in r["phi"] if s["type"] == "HOSPITAL")
            for h in REAL_HOSPITALS:
                if h in outside or h in hosp_vals:
                    problems.append((rid, f"real hospital name: {h}"))
            # 4. Leftover placeholders / simplified Chinese.
            if re.search(r"\{[A-Z]+\}", r["text"]):
                problems.append((rid, "unfilled placeholder"))
            # Fake surnames (游, 范...) live inside spans, so check only the unlabelled text;
            # 台 vs 臺 is an accepted Taiwanese variant.
            conv = S2TW.convert(outside)
            # 台/臺 and 症/癥 are accepted Taiwanese forms (OpenCC prefers 臺 and 癥).
            diff = [a for a, b in zip(outside, conv) if a != b and a not in "台症"]
            if diff:
                problems.append((rid, f"simplified/variant chars: {''.join(diff[:5])}"))
    print(f"records={n} problems={len(problems)}")
    for p in problems[:80]:
        print(*p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data/release/hf"))
