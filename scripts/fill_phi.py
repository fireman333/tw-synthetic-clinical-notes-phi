"""Stage 2: fill skeleton placeholders with fake values and record exact offsets.

Usage: python synth/fill_phi.py --in data/skeletons.jsonl --out data/notes.jsonl
"""
import argparse
import json
import random
import re
from datetime import date, timedelta

from faker import Faker

PH_RE = re.compile(r"\{([A-Z]+)\}")
ID_LETTERS = "ABCDEFGHJKLMNPQRSTUVXYWZIO"  # official order used by the checksum

# Deliberately fanciful prefixes; avoid names of real Taiwanese hospitals (仁濟, 康寧, 博愛, 惠民... are real).
HOSP_PREFIX = ["青嵐", "雲杉", "星湖", "月橋", "楓林", "曉山", "嵐川", "霽光", "澄波", "松濤", "白鷺", "銀杏"]
HOSP_SUFFIX = ["綜合醫院", "紀念醫院", "醫院", "診所", "聯合診所", "醫學中心"]


def tw_id_checksum_ok(s):
    n = ID_LETTERS.index(s[0]) + 10
    digits = [n // 10, n % 10] + [int(c) for c in s[1:]]
    weights = [1, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1]
    return sum(d * w for d, w in zip(digits, weights)) % 10 == 0


# Public figures whose names should never appear as fake patients/doctors.
CELEBRITY_NAMES = {
    "林俊傑", "周杰倫", "蔡依林", "張惠妹", "林志玲", "王力宏", "五月天", "蕭敬騰", "楊丞琳", "羅志祥",
    "蔡英文", "賴清德", "馬英九", "陳水扁", "李登輝", "柯文哲", "侯友宜", "韓國瑜", "朱立倫", "蔣萬安",
    "郭台銘", "張忠謀", "黃仁勳", "王建民", "陳金鋒", "戴資穎", "林書豪", "曾雅妮", "李安", "侯孝賢",
}


class Faker_TW:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.fk = Faker("zh_TW")
        self.fk.seed_instance(seed)
        # Per-note memory so a repeated {NAME} usually refers to the same person.
        self.memo = {}

    def new_note(self, skeleton=""):
        self.memo = {}
        head = skeleton[:200]
        self.gender = "F" if re.search(r"女|female|\bF\b", head) else "M" if re.search(r"男|male|\bM\b", head) else None
        # One anchor date per note; later {DATE}s move forward in time, mostly in one format.
        self.anchor = date(2019, 1, 1) + timedelta(days=self.rng.randint(0, 2200))
        self.date_fmt = self.rng.choice(["iso", "slash", "roc", "md", "en"])

    def value(self, kind):
        if kind in ("DATE", "RELATIVE"):
            return getattr(self, f"_{kind.lower()}")()
        # 70% reuse the first value of this kind in the note (same patient, same doctor).
        if kind in self.memo and self.rng.random() < 0.7:
            return self.memo[kind]
        v = getattr(self, f"_{kind.lower()}")()
        self.memo.setdefault(kind, v)
        return v

    def _name(self):
        while True:
            if self.gender == "F":
                v = self.fk.name_female()
            elif self.gender == "M":
                v = self.fk.name_male()
            else:
                v = self.fk.name()
            if v not in CELEBRITY_NAMES:
                return v

    def _relative(self):
        # Family member: a different person from the patient.
        v = self.fk.name()
        while v == self.memo.get("NAME") or v in CELEBRITY_NAMES:
            v = self.fk.name()
        return v

    def _doctor(self):
        v = self.fk.name()
        while v in CELEBRITY_NAMES:
            v = self.fk.name()
        return v

    def _mrn(self):
        return str(self.rng.randint(1000000, 99999999))

    def _idno(self):
        # Deliberately generate IDs that FAIL the official checksum -> cannot be a real ID.
        while True:
            s = self.rng.choice(ID_LETTERS[:24]) + self.rng.choice("12") + "".join(
                str(self.rng.randint(0, 9)) for _ in range(8))
            if not tw_id_checksum_ok(s):
                return s

    def _date(self):
        d = self.anchor
        self.anchor += timedelta(days=self.rng.choice([0, 1, 2, 3, 5, 7, 14, 28, 90]))
        fmt = self.date_fmt if self.rng.random() < 0.8 else self.rng.choice(["iso", "slash", "roc", "md", "en"])
        if fmt == "iso":
            return d.isoformat()
        if fmt == "slash":
            return f"{d.year}/{d.month:02d}/{d.day:02d}"
        if fmt == "roc":
            return f"{d.year - 1911}/{d.month:02d}/{d.day:02d}"
        if fmt == "md":
            return f"{d.month}/{d.day}"
        return d.strftime("%b %d, %Y")

    def _phone(self):
        if self.rng.random() < 0.6:
            return f"09{self.rng.randint(10, 99)}-{self.rng.randint(0, 999):03d}-{self.rng.randint(0, 999):03d}"
        return f"(0{self.rng.randint(2, 8)}) {self.rng.randint(2000, 8999)}-{self.rng.randint(0, 9999):04d}"

    def _address(self):
        return self.fk.address()

    def _hospital(self):
        return self.rng.choice(HOSP_PREFIX) + self.rng.choice(HOSP_SUFFIX)


# Placeholders that are labelled with a different PHI type.
TYPE_OF = {"RELATIVE": "NAME"}


def fill(skeleton, fk):
    fk.new_note(skeleton)
    out, spans, deid, pos = [], [], [], 0
    cursor = 0
    for m in PH_RE.finditer(skeleton):
        kind = m.group(1)
        before = skeleton[cursor:m.start()]
        out.append(before)
        deid.append(before)
        pos += len(before)
        v = fk.value(kind)
        spans.append({"type": TYPE_OF.get(kind, kind), "text": v, "start": pos, "end": pos + len(v)})
        out.append(v)
        deid.append(f"[{TYPE_OF.get(kind, kind)}]")
        pos += len(v)
        cursor = m.end()
    out.append(skeleton[cursor:])
    deid.append(skeleton[cursor:])
    text = "".join(out)
    assert all(text[s["start"]:s["end"]] == s["text"] for s in spans)
    return text, spans, "".join(deid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    fk = Faker_TW(args.seed)
    n = 0
    with open(args.inp, encoding="utf-8") as fi, open(args.out, "w", encoding="utf-8") as fo:
        for i, line in enumerate(fi):
            rec = json.loads(line)
            text, spans, deid = fill(rec["skeleton"], fk)
            fo.write(json.dumps({"id": f"syn-{i:05d}", **{k: rec[k] for k in ("specialty", "note_type", "style", "generator_model")},
                                 "text": text, "phi": spans, "deid_text": deid}, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} notes -> {args.out}")


if __name__ == "__main__":
    main()
