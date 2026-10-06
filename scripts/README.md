# scripts

Code used to build this dataset (MIT License). Run order:

1. `gen_skeletons.py` — stage 1, Gemma 4 26B-A4B (mlx-lm) writes placeholder skeletons
2. (QA keep/drop verdicts, not included)
3. `build_release.py` — fills fake PHI (`fill_phi.py`), splits by skeleton, writes `data/`, `sft/`, card
4. `privacy_check.py data/` — pre-publication privacy & integrity gate
5. `evaluate.py` — span-recall evaluation used in the tutorial

Paths inside the scripts assume the original project layout; adjust `ROOT` as needed.
