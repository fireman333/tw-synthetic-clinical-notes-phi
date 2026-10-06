---
license: cc-by-4.0
language:
- zh
- en
task_categories:
- token-classification
- text-generation
tags:
- medical
- synthetic
- de-identification
- phi
- traditional-chinese
size_categories:
- 1K<n<10K
pretty_name: Taiwan Synthetic Clinical Notes with PHI Labels
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/validation.jsonl
  - split: test
    path: data/test.jsonl
---

# tw-synthetic-clinical-notes-phi

1,000 份**完全虛構**、中英夾雜的繁體中文合成病歷。每一份都標好 PHI（個人可識別資訊）的類型、字串與字元位置，可以用來教學，或做去識別化模型的實驗。

這份資料集是教學網站「本地 LLM 冒險」（<https://med-study-rpg.com/local-llm/>）主線專案「病歷守門人」使用的資料。

## 完全虛構聲明

- 本資料集**不是從任何真實病歷衍生**，也沒有使用任何真實病人資料。
- 病歷「骨架」由開源模型生成。骨架中所有可識別資訊都用佔位符代替，例如 `{NAME}`、`{DATE}`。
- 所有 PHI 值由程式（Python + Faker 與自訂規則）從假資料填入，填入時同步記錄精確的字元位置。
- 身分證字號（IDNO）**刻意產生不通過官方檢查碼的字串**，所以不可能是真實存在的號碼。
- 醫院名稱使用虛構的名字，並排除已知的真實院名。若和現實中的院所同名，純屬巧合。

## 生成方法

### 第一段：骨架生成（LLM，本機）

- 模型：Gemma 4 26B-A4B（QAT 4-bit，Apache-2.0 授權），在本機 Apple Silicon（mlx-lm 0.32）上執行，沒有使用任何雲端 LLM API。
- 每份骨架的條件都隨機抽樣決定：
  - 科別（20 科）
  - 主題（每科 5–7 個常見問題）
  - 病人年齡與性別：與疾病相符，例如細支氣管炎限 2 歲以下
  - 病歷類型：急診、入院、出院摘要、病程紀錄、會診單、門診
  - 寫作風格：電報式、中英夾雜、敘述式
  - 長度
- 後處理分兩步：
  1. 用 OpenCC `s2tw` 做字元層級的簡轉繁。
  2. 用一張小對照表把中國用語改成台灣醫界用語，例如前列腺 → 攝護腺。
- 自動剔除以下情形：
  - 夾帶思考過程或 markdown
  - 有 LaTeX 或外文雜字
  - 佔位符種類不合規範，或少於 3 個
  - 移除佔位符後，仍殘留疑似電話、身分證、日期、長數字、院名
  - 長度過短

### 第二段：品質審查（只判定保留或剔除）

- 每一份骨架都經過逐份審查，看五個面向：
  - 醫學內容是否合理、前後一致，例如左右側、用藥與診斷、劑量、年齡與疾病
  - 有沒有簡體字或中國用語
  - 有沒有疑似真實資訊
  - 格式是否正確
  - 佔位符用法是否正確
- 有重大醫學錯誤、疑似洩漏或嚴重格式問題的骨架一律剔除。
- 審查由 Claude（Anthropic）擔任，**只輸出保留或剔除的判定，不改寫任何內容**。資料集裡的所有文字都來自上述開源模型與程式。
- 審查共檢視 1,350 份，保留 1,134 份；本資料集再從保留的骨架中隨機抽出 1,000 份。

### 第三段：PHI 填入（程式）

- 用 Python 把佔位符換成假值。同一份病歷內：
  - 日期依時間先後遞增，大多使用同一種格式
  - 病人與醫師姓名有一定機率重複出現
  - 家屬是另外一個人
- 同步寫入每個 PHI 的 `start` / `end` 字元位置，並產生遮蔽版本 `deid_text`。
- **以骨架為單位**切分 train / validation / test（約 70 / 10 / 20），每個骨架只出現在一個 split，避免同一份骨架同時出現在訓練與測試中。

### 發布前檢查（`privacy_check.py`）

全部 1,000 份都通過以下檢查：
- 每個標註的字元位置都和原文一致。
- 所有 IDNO 都不通過官方檢查碼。
- `deid_text` 不殘留任何 PHI 字串。
- 標註範圍以外，沒有身分證、手機、市話、完整日期、7 位以上長數字、email、門牌地址樣式的字串。
- 標註範圍以外與假院名中，都沒有已知的真實院名。
- 沒有未填入的佔位符，標註範圍以外也沒有簡體字。
- 沒有重複的 id 或本文。

## 檔案

```
data/train.jsonl        700 份
data/validation.jsonl   100 份
data/test.jsonl         200 份
sft/                    同一份資料轉成 chat 格式（mlx-lm / Unsloth 可直接讀）
notebooks/              Colab notebook（Gemma 4 E2B + Unsloth）
scripts/                產生本資料集的程式（MIT 授權）
stats.json              統計數字
LICENSE                 CC BY 4.0（資料）
```

在 Python 讀取：

```python
from datasets import load_dataset
base = "https://raw.githubusercontent.com/fireman333/tw-synthetic-clinical-notes-phi/main/data"
ds = load_dataset("json", data_files={s: f"{base}/{s}.jsonl" for s in ("train", "validation", "test")})
```

## 欄位說明

| 欄位 | 型別 | 說明 |
|---|---|---|
| `id` | string | 唯一編號 |
| `specialty` | string | 科別 |
| `topic` | string | 主要問題（生成時指定的主題） |
| `note_type` | string | 病歷類型 |
| `style` | string | 寫作風格：`telegraphic` / `mixed` / `narrative` |
| `text` | string | 病歷本文（含填入的假 PHI） |
| `phi` | list of {type, text, start, end} | PHI 標註；`start`、`end` 為 `text` 內的字元位置 |
| `deid_text` | string | 以 `[TYPE]` 標籤遮蔽 PHI 後的版本 |

## PHI 類型

| 類型 | 說明 | 本資料集筆數 |
|---|---|---|
| NAME | 病人或家屬姓名（骨架中的 `{RELATIVE}` 也標為 NAME） | 1,605 |
| DOCTOR | 臨床人員姓名 | 1,129 |
| MRN | 病歷號（7–8 位數字） | 644 |
| IDNO | 身分證字號（刻意不通過檢查碼） | 447 |
| DATE | 日期（ISO、斜線、民國年、月/日、英文月份等寫法） | 1,875 |
| PHONE | 電話（手機與市話） | 338 |
| ADDRESS | 街道地址（Faker 假地址） | 231 |
| HOSPITAL | 醫院或診所名稱（虛構） | 644 |

平均每份 6.9 個 PHI，平均長度 1120.4 字元。各科別、病歷類型與寫作風格的分布見 `stats.json`。

## 已知限制

- **分布比真實病歷單純**：骨架由模型生成，版面規整，缺少真實病歷的雜訊、複製貼上痕跡與個人縮寫習慣。在這份資料上的分數，不代表在真實病歷上的表現。
- **醫學內容可能仍有錯**：雖然經過審查並剔除重大錯誤，仍保留了不少小瑕疵，例如劑量偏離常見範圍、用詞不夠精準，也沒有由臨床專業人員逐份審閱。**不宜當作醫學知識來源。**
- **假姓名、假地址、假電話可能和真實存在的人、門牌或門號巧合相同**：這些值都由 Faker 與隨機規則產生，和任何真實就醫紀錄無關；姓名已排除常見公眾人物。**電話號碼請勿撥打。**
- **年齡不在標註範圍內**：本資料集不標註 AGE。HIPAA Safe Harbor 把 90 歲以上的年齡視為識別項，若你的用途需要比照，請自行處理。
- **未標註的日期型文字**：骨架裡可能殘留單獨的年份，例如「s/p CABG 2018」。這不屬於 DATE 標註，也不在 `phi` 內。
- 標註由程式依佔位符產生。骨架中若有沒用佔位符寫出的識別性文字，可能沒有被標註。品質檢查已盡量攔截，但無法保證完全沒有。

## 預期用途與不宜的用途

**預期用途**
- 教學：示範如何微調小型語言模型，找出病歷中的 PHI。
- 在合成資料上做去識別化流程與評估指標的實驗與除錯。

**不宜的用途**
- 用來宣稱某個去識別化系統已可臨床使用。在這份資料上的表現，不能作為處理真實病歷的依據。
- 當作醫學知識、診斷或治療建議的來源。
- 嘗試用這份資料還原或推測任何真實個人。

若要處理真實病歷，建議先取得院方核准與 IRB（人體研究倫理審查）同意，並在合規環境中進行。

## 回報問題

發現疑似真實資訊、標註錯誤或醫學錯誤，請在本 repo 開 Issue。

## 作者與引用

由 WLK（醫學系畢業）製作，屬於教學網站「本地 LLM 冒險」的一部分。

```bibtex
@misc{tw_synthetic_clinical_notes_phi_2026,
  title  = {Taiwan Synthetic Clinical Notes with PHI Labels},
  author = {WLK},
  year   = {2026},
  url    = {https://github.com/fireman333/tw-synthetic-clinical-notes-phi}
}
```

## 授權

資料集以 CC BY 4.0 釋出。骨架由 Gemma 4 26B-A4B（Apache-2.0）生成。
