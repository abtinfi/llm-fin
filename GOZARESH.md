# پیاده‌سازی و ارزیابی چهار Aim پروپوزال

**Model:** BioMistral-7B · bf16 · بدون quantization  **Hardware:** یک RTX 4090 (GPU 0)
**Decoding:** greedy، تک seed  **Ground truth:** programmatic — بدون LLM-as-judge

گزارش کامل کاری که تا امروز انجام شده: یک ablation matrix که اثر هر contribution را در برابر baseline می‌سنجد، روی دو benchmark با counterfactual pairs — یکی synthetic و یکی روی clinical notes واقعی — به‌علاوهٔ پیاده‌سازی هر چهار Aim: SAE برای discovery، probing و activation patching برای mechanism، Constraint-Aware Layer آموزش‌دیده برای control، و conformal prediction برای validation. هر سه RQ پروپوزال پاسخ گرفته‌اند، دو تای آن‌ها با پاسخ منفی.

---

## 1. چه چیزی پیاده شد

کار با کوچک‌ترین dataset و ساده‌ترین task شروع شد و یک baseline ساخته شد که هیچ‌کدام از contributionها را ندارد؛ بعد هر contribution یکی‌یکی اضافه شد تا اثر جداگانه‌اش دیده شود. همان ablation matrix بخش 4.6 پروپوزال، منتها با ردیف‌های اضافه که در بخش 3 توضیح داده شده. discovery pipeline پروپوزال هم کامل پیاده شده است:

`Clinical prompt → Hidden states → SAE features → Concept probing → Causal test → Intervention adapter → UQ / abstention`

### وضعیت چهار Aim

| Aim | وضعیت | چه چیزی پیاده شد |
|---|---|---|
| **Aim 1** — Discovery | پیاده شد | Sparse Autoencoder روی residual stream (هر دو architecture: TopK و JumpReLU)، معیار FIS، و concept probing روی feature activations |
| **Aim 2** — Mechanism | پیاده شد | Linear probing لایه‌به‌لایه با null control، activation patching و ACE per layer، و feature knock-out با matched random control |
| **Aim 3** — Control | پیاده شد | Constraint-Aware Layer به‌صورت residual adapter با هر چهار term تابع هدف، روی base model کاملاً frozen. به‌علاوهٔ Symbolic Gate روی constraint graph |
| **Aim 4** — Validation | پیاده شد | Counterfactual benchmark با Causal Consistency در سطح pair، Adaptive Conformal Inference برای abstention threshold، و conditional coverage روی subgroupها |

### پاسخ به سه Research Question پروپوزال

| RQ | پاسخ | شاهد |
|---|---|---|
| **RQ1** — آیا sparse featureها به biomedical concepts نگاشت می‌شوند؟ | **بله** | SAE featureهایی با selectivity بالا: QT interval F1 = 0.703، creatinine 0.603، pregnancy 0.615 |
| **RQ2** — آیا intervention روی این featureها تغییر رفتاری پیش‌بینی‌پذیر می‌دهد؟ | **خیر** | knock-out فقط 0.002 تا 0.016 logit جابه‌جایی می‌دهد، در حالی که flip یک decision به 1 تا 5 logit نیاز دارد. اما یک adapter آموزش‌دیده روی همان representation جواب می‌دهد — بخش 4 |
| **RQ3** — آیا symbolic constraints بدون آسیب به توانایی زبانی، consistency را بالا می‌برند؟ | **بله** | Causal Consistency از 0.000 به 0.767، با perplexity از 7.1622 به 7.0775 یعنی 1.2٪ کاهش |

---

## 2. Benchmarkها — از کوچک و ساده به بزرگ و واقعی

چون معیار اصلی پروپوزال Causal Consistency است، dataset باید به‌صورت **counterfactual pair** ساخته می‌شد: برای هر بیمار دو نسخه که فقط **یک متغیر علّی** بینشان فرق دارد و label حتماً برعکس می‌شود.

- eGFR = 27 → تجویز metformin برابر **UNSAFE**
- eGFR = 60 → همان تجویز برابر **SAFE**

اگر مدل mechanism را فهمیده باشد باید **هر دو arm** را درست بگوید. اگر فقط pattern matching کند، یکی را می‌زند و یکی را نه. به همین دلیل Causal Consistency در **سطح pair** شمرده می‌شود، نه per-item: مدلی که به همه‌چیز UNSAFE بگوید per-arm پنجاه درصد می‌گیرد ولی pair-level صفر.

| پله | چه چیزی | حجم |
|---|---|---|
| 1 | Synthetic vignette، متغیر علّی مستقیماً در متن نوشته شده، 8 rule family | 128 items / 64 pairs |
| 2 | همان، hardened: مقادیر near-threshold، بیان implicit در نیمی از templateها، سه distractor lab در هر case | 128 items |
| 3 | دو rule family کاملاً unseen برای سنجش generalization | 32 items / 16 pairs |
| 4 | **Clinical notes واقعی** از MedCalc-Bench — کمیت تصمیم باید از روی lab value **محاسبه** شود، نه خوانده | 240 items / 120 pairs |
| 5 | MedMCQA + MedQA + PubMedQA برای سنجش uncertainty estimator | 6,456 items |
| 6 | Control pairs برای اعتبارسنجی **خودِ metric** | 8,000 items |

در پلهٔ 4، یک arm هر pair یک **note واقعی و دست‌نخورده** است و arm دیگر همان note با **یک عدد** تغییریافته، طوری که مقدار محاسبه‌شده از threshold رد شود. بقیهٔ متن byte-به-byte یکسان است، پس هر تغییری در جواب مدل فقط به همان یک edit قابل نسبت‌دادن است. Thresholdها از **FDA label** نقل شده‌اند: متفورمین eGFR < 30 و اوندانسترون QTc > 500.

---

## 3. Ablation matrix — اثر هر contribution در برابر baseline

جدول صرفاً cumulative نیست. یک ladder تجمعی نمی‌تواند اثر را به یک مؤلفه نسبت دهد: روی clinical notes واقعی، RAG خودش Causal Consistency را **خراب** می‌کند، پس «سهم gate» اگر روی RAG سوار شود از یک نقطهٔ خراب اندازه‌گیری شده. برای همین ردیف‌های 5 تا 7 هر contribution را **مستقیماً روی base model** می‌گذارند و McNemar هر ردیف در برابر **خود baseline** گرفته می‌شود.

### Real clinical notes — held-out split
خانوادهٔ QT، 60 items / 30 pairs، هرگز training ندیده

| Variant | Accuracy | Δ Acc | Causal Consistency | Δ CC | Violation | Coverage | McNemar vs base |
|---|---|---|---|---|---|---|---|
| (1) Base LLM | 0.500 | — | 0.000 | — | 1.000 | 1.000 | — |
| (2) + RAG | 0.500 | +0.000 | 0.000 | +0.000 | 0.000 | 1.000 | p = 1 |
| (3) + Symbolic Gate | 1.000 | +0.500 | 1.000 | +1.000 | 0.000 | 1.000 | 1.9e-09 |
| (4) + UQ Engine | 1.000 | +0.500 | 1.000 | +1.000 | 0.000 | 1.000 | 1.9e-09 |
| (5) Base + Gate تنها | 1.000 | +0.500 | 1.000 | +1.000 | 0.000 | 1.000 | 1.9e-09 |
| (6) Base + UQ تنها | 0.000 | -0.500 | 0.000 | +0.000 | 0.000 | 0.000 | 1.9e-09 |
| **(7) Base + Constraint Layer تنها** | **0.883** | **+0.383** | **0.767** | **+0.767** | 0.133 | 1.000 | **1.5e-05** |
| (8) هر چهار contribution | 1.000 | +0.500 | 1.000 | +1.000 | 0.000 | 1.000 | 1.9e-09 |

### Synthetic vignettes — test split
128 items / 64 pairs

| Variant | Accuracy | Δ Acc | CC | Δ CC | Violation | Coverage | Gate fired |
|---|---|---|---|---|---|---|---|
| (1) Base LLM | 0.602 | — | 0.219 | — | 0.125 | 1.000 | 0.000 |
| (2) + RAG | 0.648 | +0.047 | 0.312 | +0.094 | 0.141 | 1.000 | 0.000 |
| (3) + Symbolic Gate | 0.836 | +0.234 | 0.672 | +0.453 | 0.125 | 1.000 | 0.562 |
| (4) + UQ Engine | 0.562 | -0.039 | 0.562 | +0.344 | 0.000 | 0.570 | 0.562 |
| (5) Base + Gate تنها | 0.805 | +0.203 | 0.609 | +0.391 | 0.062 | 1.000 | 0.562 |
| (6) Base + UQ تنها | 0.000 | -0.602 | 0.000 | -0.219 | 0.000 | 0.000 | 0.000 |
| (7) Base + Constraint Layer | 0.703 | +0.102 | 0.422 | +0.203 | 0.344 | 1.000 | 0.000 |
| (8) هر چهار contribution | 0.562 | -0.039 | 0.562 | +0.344 | 0.000 | 0.586 | 0.562 |

روی real notes، test split با 180 items: base CC = 0.011 ← RAG = 0.000 ← Gate = 0.989 با p = 6.5e-27، 88 improvement و 0 regression. همهٔ رقم‌ها با bootstrap CI و coverage گزارش می‌شوند؛ چون decoding قطعی است، به‌جای standard deviation روی seed، **CI روی items** داده می‌شود.

---

## 4. یافته‌ها

### 1 — Constraint-Aware Layer کار می‌کند، و جایی که کار نمی‌کند از قبل پیش‌بینی شده بود

یک **rank-32 adapter** با 262 هزار parameter روی layer 30، با base model کاملاً frozen، آموزش روی 55 pair: Causal Consistency روی 30 pair دیده‌نشده از **0.000 به 0.767** رفت، با CI برابر 0.600 تا 0.900. Control با shuffled labels به 0.200 می‌رسد که **دقیقاً سطح chance** است — یک predictor تصادفی در سطح pair عدد 0.25 می‌گیرد — و حتی نمی‌تواند training set خودش را fit کند.

روی renal family همان adapter با 140 pair آموزش **هیچ چیز یاد نگرفت**، حتی روی training set خودش با decision accuracy برابر 0.564. این شکست **قبل از training** پیش‌بینی شده بود: linear probe نشان داده بود QTc از residual stream قابل decode است با pair-CC برابر 0.976 در برابر null برابر 0.200، ولی creatinine نیست با 0.042 در برابر null برابر 0.033.

> Constraint layer فقط جایی کار می‌کند که مدل کمیت تصمیم را از قبل represent کرده و صرفاً آن را read out نمی‌کند — و اینکه در کدام حالت هستیم با یک probe ارزان و از قبل قابل سنجش است.

**RQ3:** WikiText-2 perplexity از 7.1622 به 7.0775 یعنی 1.2٪ کاهش. این دو عدد باید همیشه کنار هم نقل شوند — layerی که به همه‌چیز UNSAFE بگوید هم CC را بالا می‌برد و در عوض مدل را نابود می‌کند.

### 2 — SAE فیچرهای مفهومی پیدا می‌کند، ولی آن featureها decision را نمی‌رانند

SAE از نوع TopK روی 150,000 token از clinical notes واقعی، با 16,384 feature و خطای بازسازی FVU = 0.067. featureهای تک‌مفهومی واضح: QT interval با F1 = 0.703، pregnancy با 0.615، creatinine با 0.603. معماری JumpReLU — همان fallback که پروپوزال نام برده — به‌وضوح ضعیف‌تر بود با FVU = 0.131 و بهترین F1 برابر 0.363، و انتخاب architecture بر همین مبنا انجام شد.

اما **feature knock-out**: وقتی سهم یک feature را از residual stream کم می‌کنیم، decision logit فقط 0.002 تا 0.016 جابه‌جا می‌شود، در حالی که flip یک decision به 1 تا 5 logit نیاز دارد — حدود **100 برابر کمتر**. دو تا از هشت feature حتی کمتر از matched random control خودشان اثر داشتند.

این **مستقلاً** همان نتیجه‌ای است که activation patching داده بود، یعنی حدود 0.002 logit. دو intervention کاملاً متفاوت — یکی overwrite یک position، یکی حذف یک learned feature از همه‌جا — به یک نقطه رسیدند، پس دیگر نمی‌شود گفت artifact انتخاب محل مداخله بوده:

> کمیت‌ها represent می‌شوند، sparse dictionary پیدایشان می‌کند، و decision آن‌ها را نمی‌خواند.

این دقیقاً همان gap است که Aim 3 برایش وجود دارد.

### 3 — RAG کافی نیست، و علتش failure در retrieval نیست

اول شک کردیم retrieval خراب باشد. روی real notes، **hit@3 برابر 99.5٪** در test و **100٪** در held-out است — یعنی سند حاوی threshold عملاً برای تک‌تک موارد بازیابی شده و جلوی چشم مدل قرار گرفته. با این حال Causal Consistency روی 0.000 ماند.

بررسی موارد نشان داد چه اتفاقی می‌افتد: بعد از دیدن متن contraindication، مدل به **96٪** موارد UNSAFE می‌گوید و accuracy روی caseهای واقعاً SAFE از 0.372 به 0.037 سقوط می‌کند. یعنی بازیابیِ متنِ هشدار، یک bias سیستماتیک به سمت هشدار ایجاد می‌کند که patient data را override می‌کند. این برای یک clinical safety system یک failure mode مشخص و actionable است، نه یک null result.

### 4 — Estimator عدم‌قطعیت پروپوزال غلط است، ولی ایدهٔ آن نه

روی 6,456 item، token-level predictive entropy معادلهٔ (2) عدد AUROC = 0.525 می‌دهد با CI برابر 0.511 تا 0.539 — یعنی تقریباً بی‌اثر. همان entropy وقتی فقط روی **answer tokens** محدود شود به 0.687 می‌رسد، با Δ = +0.163 و p < 0.0001.

علت روشن است: معادلهٔ (2) entropy را روی کل vocabulary 32 هزارتایی پخش می‌کند که بیشترش دربارهٔ نحوهٔ جمله‌بندی justification است، نه اینکه کدام answer درست است. **پیشنهاد مشخص برای پروپوزال:** معادلهٔ (2) به نسخهٔ محدودشده به decision space بازنویسی شود.

**Adaptive Conformal Inference** هم پیاده شد. روی test split، threshold ثابت coverage برابر 0.570 با خطای 0.014 می‌دهد و ACI همان را به coverage برابر 0.875 می‌رساند به قیمت خطای 0.161 — یعنی trade-off بین coverage و risk حالا **قابل تنظیم** است، نه ثابت.

### 5 — Symbolic Gate و UQ Engine سر همان caseهای آسان رقابت می‌کنند

Coverage کل روی test split برابر 0.570 است، ولی تحلیل **conditional coverage** نشان می‌دهد این عدد کل ماجرا را پنهان می‌کند:

| Subgroup | n | Coverage | Selective accuracy |
|---|---|---|---|
| gate fired | 72 | 1.000 | 1.000 |
| gate declined | 56 | 0.018 | 0.000 |

شکاف coverage بین دو subgroup برابر **0.982** است. یعنی UQ engine عملاً روی همان جمعیتی «کار می‌کند» که gate قبلاً حل کرده، و روی جمعیت واقعاً سخت سکوت می‌کند. Symbolic Gate دقیقاً caseهای قابل‌تعیین را برمی‌دارد و باقی‌مانده یک جمعیت واقعاً سخت است. این یک **نکتهٔ معماری** است، نه شکست tuning — و همان چیزی است که پروپوزال با «conditional coverage روی subgroupها» وعده داده بود.

### 6 — L_uncertainty از تابع هدف تعمیم نمی‌یابد

هر چهار term تابع هدف پروپوزال یعنی `L_LM + λ1·L_ontology + λ2·L_causal + λ3·L_uncertainty` پیاده شد. جملهٔ L_ontology به‌صورت یک constraint روی **monotonicity** پیاده شد — decision margin بازوی SAFE باید بالای بازوی UNSAFE بنشیند — و کار کرد: hinge آن طی training از 0.985 به 0.015 افتاد. یعنی adapter صرفاً labelهای تکی را درست نکرد، بلکه **جهتی که threshold دیکته می‌کند** را رعایت کرد.

اما L_uncertainty — که روی نسخهٔ redacted نُت‌ها، یعنی جایی که عدد تعیین‌کننده حذف شده، entropy را maximize می‌کند — روی training set برآورده شد ولی **تعمیم نیافت**: روی redacted noteهای دیده‌نشده مدل **مطمئن‌تر** شد، از 0.626 به 0.457 نَت، در حالی که maximum برابر ln 2 = 0.693 است. یعنی confidenceای که از L_causal یاد گرفته می‌شود منتقل می‌شود، ولی **calibrated ignorance** نه. این را به‌عنوان negative result گزارش می‌کنم چون دقیقاً خلاف فرضی است که پشت یک uncertainty-aware safety layer خوابیده.

### 7 — Constraint Layer، consistency می‌خرد و safety margin می‌فروشد

این را باید صریح گفت: هم‌زمان با بالا رفتن CC، **violation rate** هم بالا می‌رود. روی QT از 1.000 به 0.133 می‌آید که بهبود بزرگی است چون base model به همه‌چیز SAFE می‌گفت، ولی روی synthetic benchmark از 0.125 به 0.344 **بدتر** می‌شود. Symbolic Gate در هر دو حالت zero error دارد.

جمع‌بندی: gate هرجا fire شود برنده است، ولی روی clinical notes واقعی فقط روی **42.7٪** موارد اصلاً می‌تواند fire کند — عمدتاً چون یک note واقعی همان lab را چند بار با مقادیر متفاوت در طول بستری ذکر می‌کند و انتخاب «کدام مقدار» نیاز به temporal reasoning دارد، نه regex بهتر. Constraint layer به regex و threshold دستی نیاز ندارد و روی آن 57٪ باقی‌مانده تنها گزینهٔ قابل امتحان است — به شرطی که هزینهٔ violation آن کنترل شود.

---

## 4.5 کارهای دور دوم (۲۰۲۶-۰۹-۰۷)

شش موردی که باقی مانده بود انجام شد. سه‌تای آن‌ها آن چیزی نبودند که اسمشان نشان می‌داد، و همان‌ها مهم‌ترین بخش این دور است.

### الف) لایهٔ Token-to-Concept — بخش ۴.۲ پروپوزال

تنها بخشی از پروپوزال بود که **هیچ کدی نداشت**. پل §4.2 تا «probing classifier» پیاده بود و همان‌جا متوقف می‌شد.

امتیاز هر توکن برای هر concept:

    a[t, c] = مجموع روی featureهای متعلق به c از  z_f(h_t) × w_f

نکتهٔ روش: `w_f` وزن **اندازه‌گیری‌شده** است — همان excess حاصل از knock-out که از قبل روی دیسک بود — نه یک پارامتر جدید. featureای که knock-out آن از کنترل تصادفی هم‌نرخ خودش بهتر نباشد، وزن صفر می‌گیرد.

**benchmark خودش ground truth می‌دهد، بدون هیچ annotation:** دو arm یک pair دقیقاً در یک مقدار علّی فرق دارند، پس توکن‌هایی که فرق می‌کنند **همان** توکن‌های تعیین‌کننده‌اند.

| split | attributor | صدک توکنِ ویرایش‌شده | concept درست |
|---|---|---|---|
| کلیوی (renal) | لایهٔ ۴.۲ | **۰٫۹۸** | **۰٫۹۰** (شانس ۰٫۲۵) |
| کلیوی | gradient × input | ۰٫۵۱ | — |
| کلیوی | تصادفی | ۰٫۵۱ | — |

یعنی پل کار می‌کند. اما faithfulness برای **همهٔ** attributorها ضعیف است، حتی برای `occlusion` که دقیق است — و این همان null بخش Aim 2 است، این بار در سطح توکن: لایه توکن تعیین‌کننده را پیدا می‌کند، ولی نشان نمی‌دهد که تصمیم مدل به آن وابسته است.

**یک اشکال در روش خودمان که پیدا و اعلام شد:** وزن‌ها با `--causal_split test` اندازه‌گیری شده بودند و آن split فقط آیتم‌های `metformin_renal` دارد. یک feature مربوط به QT روی آیتم‌هایی که تصمیمشان به QT ربطی ندارد **ذاتاً** اثر صفر می‌دهد. پس جملهٔ «مدل feature مربوط به QT ندارد» دو ادعای متفاوت را قاطی می‌کرد. `run_sae_split_weights.sh` این دو را از هم جدا می‌کند؛ تا وقتی نتیجه‌اش نیامده، در مقاله این مورد **حل‌نشده** گزارش می‌شود، نه به‌عنوان یافته.

### ب) control pair — مهم‌ترین یافتهٔ روش‌شناختی این دور

تا امروز هر عدد Causal Consistency در این پروژه یک ضعف مشترک داشت: مدلی که به **هر** تغییر prompt واکنش نشان دهد، بدون فهمیدن قاعده هم امتیاز می‌گیرد.

`data/medcalc_v2` برای اولین بار **control pair** دارد: مقدار آزمایش به اندازهٔ مشابه تغییر می‌کند ولی از threshold **رد نمی‌شود**، پس label عوض نمی‌شود.

نتیجه روی base model (۱۱۵ pair علّی + ۱۰۹ pair کنترل):

| کمیت | مقدار |
|---|---|
| نرخ flip علّی | ۰٫۰۳۵ |
| نرخ flip کاذب | ۰٫۰۳۷ |
| **discrimination** | **−۰٫۰۰۱۹** |

مدل با همان نرخ جواب را عوض می‌کند، چه حقیقت عوض شده باشد چه نشده باشد. این دقیقاً همان **−۰٫۰۱۳ [−۰٫۰۳۷, +۰٫۰۱۳]** است که قبلاً روی ۸۰۰۰ سؤال چهارگزینه‌ای دیده بودیم — حالا روی **متن بالینی واقعی** و task کاملاً متفاوت تکرار شد. دو benchmark با شکل کاملاً متفاوت یک چیز می‌گویند: counterfactual consistency به‌تنهایی شاهد استدلال بالینی نیست.

### ج) بزرگ‌کردن benchmark، و یک سقف که پیدا شد

دو خانوادهٔ جدید با threshold نقل‌شده از برچسب FDA: نیتروفورانتوئین/CrCl و آپیکسابان/Child-Pugh کلاس C. هر آیتم اصلی **بیت‌به‌بیت** بازتولید می‌شود، پس هیچ عدد منتشرشده‌ای تکان نمی‌خورد.

دو گزینهٔ دیگر رد شدند و دلیلش ثبت شد: چهار فرمول دیگر QTc شبیه ۵ برابر شدن داده به نظر می‌رسند ولی نیستند (۵۰۰ ردیف، ولی فقط **۱۰۰ note متمایز** — هر پنج ماشین‌حساب روی همان noteها)؛ و CHA2DS2-VASc و HAS-BLED فقط با **سن** پیش می‌روند، پس ویرایش counterfactual به‌جای یک متغیر، خودِ بیمار را عوض می‌کند.

**یافته‌ای که از دل extractorها بیرون آمد — نسبت پوشش gate به پیچیدگی قاعده:**

| خانواده | تعداد متغیر | پوشش gate |
|---|---|---|
| ondansetron / QTc | ۲ | ۱٫۰۰ |
| metformin / eGFR | ۳ | ۰٫۹۹ |
| nitrofurantoin / CrCl | ۵ | ۰٫۹۲ |
| apixaban / Child-Pugh | ۵ (دوتا قضاوت بالینی) | **۰٫۰۰** |

صفرِ آخری اشکال regex نیست: کلمهٔ «encephalopathy» در **هیچ‌کدام** از ۱۰۲ نوت نیامده، پس درجه‌اش با هیچ کیفیتی از regex قابل استخراج نیست. این یک سقف برای **کل ایدهٔ gateهای threshold-خوان** است، نه برای این پیاده‌سازی.

### د) چند مدل و چند لایه

`src/lm_common.py` ساخته شد. کد قبلی دو ویژگی SentencePiece را hard-code کرده بود (نشانهٔ ▁ و newline بازگشتی `<0x0A>`) و Llama-3 هیچ‌کدام را ندارد — در حالی که **هر** margin و entropy و ACE این پروژه از همان id‌ها حساب می‌شود. resolver جدید مستقل از tokenizer است و روی BioMistral **دقیقاً** همان id‌ها را می‌دهد؛ اجرای مجدد `base` فایل prediction را بیت‌به‌بیت بازتولید کرد.

OpenBioLLM اصلاً chat template ندارد؛ اگر با `[INST]` مistral می‌پیچیدیمش، مدل دوم به دلیلی بی‌ربط به علم بدتر به نظر می‌رسید.

### ه) ارزیابی انسانی — همه‌چیز جز خودِ نمره‌ها

نمونه‌گیری طبقه‌بندی‌شده، کلید پاسخ در فایل **جدا**، ترتیب متفاوت برای هر rater، ۱۰٪ آیتم تکراری برای سنجش پایایی درون‌فردی، بستهٔ HTML مستقل، و محاسبهٔ Krippendorff α و Fleiss κ.

`ingest` روی بستهٔ خالی **عمداً خطا می‌دهد** و صفر برنمی‌گرداند؛ «اندازه‌گیری نشده» با «صفر اندازه‌گیری شد» یکی نیست. اجرای آزمایشی با نمرات ساختگی همین را نشان می‌دهد: pass rate ظاهراً قابل‌قبولِ ۰٫۸۹ در کنار α = −۰٫۳۱ — یعنی آمار توافق درست گزارش می‌کند که «rater»ها نویز بوده‌اند.

یک یافتهٔ جانبی از نمونه‌گیری: **۱۰۲ پاسخ از ۹۰۰** اصلاً justification ندارند؛ مدل با وجود درخواست صریح prompt، فقط کلمهٔ تصمیم را می‌نویسد. این‌ها کنار گذاشته و **شمرده** شدند، نه اینکه به‌عنوان «توضیح» جلوی پزشک گذاشته شوند.

### و) مقاله و اصلاحیهٔ پروپوزال

`PAPER.md` از روی artifactها **تولید** می‌شود (`src/make_paper.py`)؛ اگر artifactی نباشد build شکست می‌خورد، نه اینکه جمله را بی‌صدا حذف کند. `PROPOSAL_ERRATA.md` برای هشت جای پروپوزال جملهٔ جایگزین می‌دهد — مهم‌ترینشان اینکه **SNOMED CT دو بار در پروپوزال آمده و هیچ‌جا استفاده نشده** (آنچه واقعاً استفاده می‌شود UMLS + MED-RT است)، و اینکه معادلهٔ (۲) روی ۶۴۵۶ آیتم AUROC ۰٫۵۲۵ می‌دهد در حالی که فرم محدود به توکن‌های تصمیم به ۰٫۶۸۷ می‌رسد.

### ز) دو عددِ منتشرشده که اصلاح شد

- `make_summary.py` از دایرکتوری قدیمی می‌خواند و اجرای **۴۰-pair** را گزارش می‌کرد؛ عدد درست از اجرای ۸۶-pair است: بیشینهٔ excess تزریق **۰٫۰۱۰۱ / ۰٫۰۶۴۶** به‌جای ۰٫۰۱۳۴ / ۰٫۰۸۳۳.
- کنترل تصادفی در `patching.py --mode sufficiency` از generator سراسری torch می‌آمد که هیچ‌جا seed نمی‌خورد، پس `--seed` به آن نمی‌رسید و `excess` بین دو اجرای یکسان فرق می‌کرد.

هیچ‌کدام نتیجه‌ای را برنمی‌گرداند؛ اثر تزریق همچنان حدود دو مرتبهٔ بزرگی کمتر از ۱ تا ۵ logit لازم برای flip است.

---

## 4.6 نتایج اجرای شبانه (۲۰۲۶-۰۹-۰۸)

### چند مدل — Aim 3 روی هر سه منتقل می‌شود

constraint layer روی یک خانواده آموزش می‌بیند و روی خانواده‌ای که هرگز ندیده سنجیده می‌شود:

| مدل | CC روی held-out: base → +CL |
|---|---|
| BioMistral-7B | ۰٫۰۰۰ → **۰٫۷۶۷** |
| Llama3-OpenBioLLM-8B | ۰٫۰۰۰ → **۰٫۴۶۷** |
| Mistral-7B-Instruct-v0.2 | ۰٫۰۳۳ → **۰٫۲۶۷** |

دو مدلی که pretraining زیست‌پزشکی دارند بیشترین سود را می‌برند و مدل عمومی کمترین را. ترتیبش قابل حدس بود، ولی حالا **اندازه‌گیری شده** است نه فرض‌شده. نتیجهٔ اصلی (base در حد شانس، سیستم کامل ~۰٫۹۹) روی هر سه با p < 1e-24 برقرار است.

**یک اشکال که اولین تلاش را باطل کرد:** `run_lane_a.sh` هرگز `--model_id` را به `constraint_layer.py` نمی‌داد و آن به BioMistral پیش‌فرض می‌رفت. یعنی adapter هر مدلی روی BioMistral آموزش دیده و به مدل دیگری وصل شده بود. هر سه مدل ۴۰۹۶ بُعدی‌اند، پس adapter اشتباه تمیز بارگذاری می‌شود و عدد قابل‌قبول می‌دهد؛ محافظ hidden-size نمی‌توانست بگیردش. نشانه‌اش این بود که دو مدل تا شانزده رقم اعشار عدد یکسان دادند. artifactها دور ریخته و هر دو مدل دوباره اجرا شدند، و حالا **هر artifact مدلی که تولیدش کرده را ثبت می‌کند**.

### لایه ۲۰ انتخاب خوبی نبوده

| لایه | FVU | dead | مفهوم قابل بیان |
|---|---|---|---|
| **۱۶** | **۰٫۰۲۶** | ۶٬۱۸۶ | **۷** |
| ۲۰ (فعلی) | ۰٫۰۵۱ | ۶٬۸۵۱ | ۴ |
| ۲۴ | ۰٫۰۹۷ | ۶٬۱۷۳ | ۶ |

لایهٔ ۲۰ یک بار از منحنی probe انتخاب شد و دیگر بازبینی نشد — و بدترینِ این سه است. لایهٔ ۱۶ دو برابر بهتر بازسازی می‌کند و `qt_interval`، `inr`، `pregnancy` و `renal_disease` را هم بیان می‌کند.

پس **هر عدد §4.2 یک کران پایین است**: لایهٔ attribution فقط مفهومی را می‌تواند نام ببرد که دیکشنری برایش feature دارد، و دیکشنری‌ای که گرفته از کم‌بیان‌ترین لایه می‌آید.

### پایداری seed

سه دیکشنری از همان activationها با seedهای مختلف. index featureها بین اجراها بی‌معناست، پس مقایسه روی زیرفضا و انتخاب مفهوم انجام شد:

| A | B | mean max cos | Jaccard مفهوم |
|---|---|---|---|
| seed0 | seed1 | ۰٫۵۶ | ۱٫۰۰ |
| seed0 | seed2 | ۰٫۵۳ | ۰٫۷۵ |

دو نیمه با هم مخالف‌اند و **همین نتیجه است**: اینکه دیکشنری چه *مفهوم‌هایی* انتخاب می‌کند نسبتاً پایدار است؛ اینکه چه *جهت‌هایی* پیدا می‌کند نیست (~۰٫۵۵ کسینوس). برای ادعاهای این پروژه که همه در سطح مفهوم‌اند این جهتِ درست است، ولی هشداری است برای هر حرفی دربارهٔ یک feature منفرد: `#14294` واقعیتی دربارهٔ یک اجرای آموزش است، نه دربارهٔ مدل.

### confound مربوط به QT — پس گرفته شد

خوانش قبلی («مدل feature مربوط به QT ندارد») **آرتیفکت اندازه‌گیری خودمان بود**. وزن‌ها روی split کلیوی سنجیده شده بودند که QT در آن ذاتاً نمی‌تواند تصمیم را تکان دهد:

| وزن‌ها روی | qt_interval قابل بیان؟ | concept pointing | صدک توکن |
|---|---|---|---|
| split کلیوی | خیر | ۰٫۰۰۰ | ۰٫۴۴۶ |
| split QT | **بله** | **۱٫۰۰۰** | **۰٫۸۵۱** |

لایهٔ §4.2 روی **هر دو** خانواده کار می‌کند.

---

## 5. محدودیت‌هایی که باید شفاف گفته شود

1. **Accuracy مربوط به Symbolic Gate تا حدی circular است.** gate همان rule و thresholdی را evaluate می‌کند که labelها از آن ساخته شده‌اند، و روی benchmark واقعی همان extractorی که dataset را filter می‌کند داخل gate هم هست. عدد قابل نقل، **coverage 42.7٪** است نه accuracy برابر 0.989. جدول هر دو را نشان می‌دهد.

2. **SAE در مقیاس pilot است.** 150 هزار token از یک corpus باریک، در برابر صدها میلیون token در کارهای مرجع؛ 39٪ featureها dead هستند. نبودِ یک concept اینجا دلیل نبودنش در مدل نیست.

3. **S_human از FIS اندازه‌گیری نشده** چون expert annotator نداریم؛ وزنش صفر گذاشته شده و صریح اعلام شده. در نتیجه معیار «30٪ عبور از expert validation» در Aim 1 با وضعیت فعلی **قابل ارزیابی نیست** — یا باید به معیاری قابل سنجش بازنویسی شود، یا برای یک panel از متخصص بودجه دیده شود.

4. **یک family، 55 training pair، یک seed.** عدد 0.767 روی 30 pair است و generalization آن within-family است، نه cross-family.

5. **Arm دوم هر pair واقعی، synthetic است** — یک note واقعی با یک عدد تغییریافته. مقدارش physiologically plausible و سازگار با بقیهٔ note است، ولی بیمار observed نیست.

6. **WikiText-2 perplexity یک proxy ضعیف** برای clinical language ability است؛ perplexity روی متن پزشکی تست بهتری بود.

---

## 6. قدم بعدی

1. کنترل هزینهٔ violation در constraint layer با یک term نامتقارن در تابع هدف — خطای «UNSAFE را SAFE خواندن» باید گران‌تر شود.
2. تست L_uncertainty با وزن بزرگ‌تر و redaction روی held-out notes، تا معلوم شود عدم‌تعمیم ذاتی است یا مسئلهٔ tuning.
3. SAE روی چند layer و با corpus بزرگ‌تر، تا ادعای «چنین featureای وجود ندارد» قابل دفاع شود.
4. Thresholdهای 8 خانوادهٔ synthetic هم مثل benchmark واقعی از openFDA نقل شوند.
5. MIMIC-IV به‌عنوان پلهٔ بعدی ladder: discharge noteهای واقعی، بدون فیلترِ «قابل استخراج بودن».

---

*همهٔ اعداد این گزارش مستقیماً از فایل‌های نتیجه تولید شده‌اند (`src/make_summary.py`)، نه دستی از روی log. جدول‌ها در `results/table_*.md`، گزارش تفصیلی Aim 1 در `results/aim1_sae.md`، Aim 3 در `results/aim3_constraint_layer.md`، و coverage در `results/uq_coverage_*.md`.*
