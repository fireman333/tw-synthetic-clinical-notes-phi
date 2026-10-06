"""Shared spec for the synthetic Taiwanese clinical-note generator.

Stage 1 (gen_skeletons.py): an LLM writes a note skeleton that uses placeholders
instead of any identifier. Stage 2 (fill_phi.py): Python fills placeholders with
fake values and records exact character offsets -> gold PHI labels for free.
"""

# Placeholder -> PHI type. Order matters only for documentation.
PHI_TYPES = {
    "NAME": "patient name",
    "RELATIVE": "family member name (labelled as NAME)",
    "DOCTOR": "clinician name",
    "MRN": "medical record number (病歷號)",
    "IDNO": "Taiwan national ID (身分證字號)",
    "DATE": "any calendar date",
    "PHONE": "phone number",
    "ADDRESS": "street address",
    "HOSPITAL": "hospital / clinic name",
}

SPECIALTIES = [
    "心臟內科", "胸腔內科", "腸胃內科", "腎臟內科", "內分泌科", "感染科", "血液腫瘤科",
    "神經內科", "一般外科", "骨科", "神經外科", "泌尿科", "婦產科", "小兒科",
    "急診醫學科", "家庭醫學科", "精神科", "皮膚科", "眼科", "耳鼻喉科",
]

# Topic pools keep the dataset from collapsing onto a few templates (STEMI, CAP, AKI...).
TOPICS = {
    "心臟內科": ["NSTEMI", "心房顫動合併快速心室反應", "心衰竭急性惡化", "高血壓急症", "感染性心內膜炎", "完全房室傳導阻滯", "主動脈瓣狹窄"],
    "胸腔內科": ["COPD 急性惡化", "氣喘急性發作", "肺栓塞", "肋膜積液", "肺結核", "自發性氣胸", "間質性肺病"],
    "腸胃內科": ["上消化道出血（消化性潰瘍）", "急性胰臟炎", "肝硬化合併腹水", "膽管炎", "潰瘍性結腸炎", "B 型肝炎急性發作", "下消化道出血（憩室）"],
    "腎臟內科": ["急性腎損傷（脫水）", "高血鉀", "腎病症候群", "末期腎病血液透析通路阻塞", "低血鈉", "腎盂腎炎"],
    "內分泌科": ["糖尿病酮酸血症", "高滲透壓高血糖狀態", "低血糖", "甲狀腺風暴", "腎上腺功能不全", "高血鈣"],
    "感染科": ["蜂窩性組織炎", "菌血症（MRSA）", "登革熱", "恙蟲病", "帶狀疱疹", "泌尿道感染（ESBL）"],
    "血液腫瘤科": ["發燒性嗜中性球低下", "腫瘤溶解症候群", "缺鐵性貧血", "免疫性血小板低下", "多發性骨髓瘤", "化療後黏膜炎"],
    "神經內科": ["急性缺血性中風（右側 MCA）", "急性缺血性中風（左側 MCA）", "癲癇重積", "重症肌無力惡化", "Guillain-Barré 症候群", "細菌性腦膜炎"],
    "一般外科": ["急性闌尾炎", "急性膽囊炎", "小腸阻塞（沾黏）", "腹股溝疝氣嵌頓", "消化性潰瘍穿孔", "乳房腫塊切片"],
    "骨科": ["股骨頸骨折", "橈骨遠端骨折", "退化性膝關節炎行全膝置換", "腰椎間盤突出", "化膿性關節炎", "踝關節骨折"],
    "神經外科": ["急性硬腦膜下出血", "慢性硬腦膜下出血", "蜘蛛膜下腔出血（動脈瘤）", "腦瘤術後", "水腦症放置引流管"],
    "泌尿科": ["輸尿管結石", "良性攝護腺肥大合併尿滯留", "睪丸扭轉", "膀胱癌血尿", "急性攝護腺炎"],
    "婦產科": ["子宮外孕", "子癇前症", "產後出血", "卵巢囊腫扭轉", "早期破水", "妊娠劇吐"],
    "小兒科": ["細支氣管炎（RSV）", "川崎病", "熱性痙攣", "急性腸胃炎合併脫水", "手足口病", "新生兒黃疸"],
    "急診醫學科": ["一氧化碳中毒", "過敏性休克", "多重創傷（機車事故）", "藥物過量（benzodiazepine）", "熱衰竭", "上消化道異物"],
    "家庭醫學科": ["高血壓追蹤", "第二型糖尿病追蹤", "高血脂症", "戒菸門診", "失眠", "成人健檢異常追蹤"],
    "精神科": ["重鬱症住院", "雙極性疾患躁期", "思覺失調症急性發作", "酒精戒斷", "恐慌症", "譫妄會診"],
    "皮膚科": ["Stevens-Johnson 症候群", "乾癬", "異位性皮膚炎", "疥瘡", "蕁麻疹"],
    "眼科": ["急性隅角閉鎖性青光眼", "視網膜剝離", "白內障手術", "角膜潰瘍", "糖尿病視網膜病變"],
    "耳鼻喉科": ["扁桃腺周圍膿瘍", "突發性耳聾", "鼻出血", "急性會厭炎", "鼻咽癌追蹤"],
}

NOTE_TYPES = {
    "急診病歷": "Emergency department note: CC, PI, PH, PE, lab, impression, plan",
    "入院病歷": "Admission note: CC, PI, PH, allergy, PE, assessment, plan",
    "出院摘要": "Discharge summary: diagnosis, hospital course, discharge meds, follow-up",
    "病程紀錄": "Progress note in SOAP format",
    "會診單": "Consultation request and reply between two departments",
    "門診紀錄": "Outpatient visit note: S/O/A/P, short",
}

STYLES = {
    "telegraphic": "電報式，大量英文縮寫與片語（例如 'BP 160/90, s/p PCI, r/o ACS'），很少完整句子",
    "mixed": "中英夾雜，中文敘述加英文醫學術語，台灣住院醫師常見寫法",
    "narrative": "較完整的中文句子，英文只用在診斷、藥名與檢驗",
}

SKELETON_PROMPT = """你是台灣醫學中心的住院醫師，請寫一份「完全虛構」的{note_type}（{note_type_desc}），科別：{specialty}。
病人：{age} 歲{sex}，主要問題：{topic}。
寫作風格：{style_desc}。長度約 {length} 個字。

最重要的規則——所有可識別個人或機構的資訊都不可以寫出真實內容，一律改用下列佔位符（含大括號，原樣輸出）：
{{NAME}} 病人姓名
{{RELATIVE}} 家屬或聯絡人姓名
{{DOCTOR}} 醫師姓名
{{MRN}} 病歷號
{{IDNO}} 身分證字號
{{DATE}} 任何日期（例如入院日、手術日、回診日）
{{PHONE}} 電話
{{ADDRESS}} 地址
{{HOSPITAL}} 醫院或診所名稱

要求：
1. 至少使用 {min_ph} 個佔位符，其中必須包含 {{NAME}}、{{DATE}}，並自然地使用 {must} 。同一種佔位符可以重複出現。
2. 年齡、性別、症狀、檢驗數值、藥名、劑量照常寫（這些不是佔位符）。醫學內容要合理一致：診斷、檢查、檢驗與用藥必須互相吻合，用藥須符合該診斷的常規治療。
   - 左右側、病灶側與症狀側必須對應（例如左側 MCA 梗塞 → 右側無力）。
   - 不可同時開兩種同類藥物（例如兩種 SSRI、兩種 beta-blocker），不可編造藥名或診斷。
   - 劑量與輸液速率要符合年齡體重的常規範圍。
   - 使用台灣醫界用語（攝護腺、敗血症、周邊血、循環、軟體、資訊）。
3. 除了佔位符之外，不可以出現任何人名、醫院名、電話號碼、病歷號或具體日期。
4. 佔位符只放在它代表的位置（例如 {{NAME}} 只指病人本人），不要把多個佔位符堆在文末，不要寫成 LaTeX 或其他格式。
5. 只輸出病歷本文，不要標題說明、不要 markdown（不要 # 或 **）、不要解釋、不要「[Your Name]」這類範本文字。"""

# Mainland-China terms -> Taiwanese clinical usage (applied after OpenCC).
TW_TERMS = {
    "前列腺": "攝護腺", "敗血癥": "敗血症", "低蛋白血癥": "低白蛋白血症", "外周血": "周邊血",
    "迴圈": "循環", "癥狀": "症狀", "併發癥": "併發症", "高血壓病": "高血壓", "冠心病": "冠狀動脈疾病",
    "晶狀體": "水晶體", "血癥": "血症",
    # Reversals of OpenCC s2twp phrase over-conversion seen in QA (局部->區域性, 分區->分割槽, 支持->支援...).
    "區域性": "局部", "割槽": "區", "互動動": "交互動", "支援": "支持", "物件": "對象",
    "言語賓士": "言語奔馳", "思緒賓士": "思緒奔馳", "巡影片率": "巡視頻率", "異物透過": "異物通過",
}
