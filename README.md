# NHANES 2021–2023 — Cardiometabolic Risk Screening (Leakage-Free)

A machine learning project that screens for **4 cardiometabolic conditions** using only **low-cost, easy-to-collect inputs** — no expensive blood tests required as inputs. The models flag high-risk patients so clinicians know who should be sent for confirmatory lab testing.

---

## What this project does

This is a **screening tool**, not a diagnostic tool. The distinction matters:

- **Diagnosis** = the actual lab test (e.g. HbA1c confirms diabetes)
- **Screening** = predicting *who is likely at risk* from cheap, fast data, so the expensive tests are ordered only for those who need them

**The workflow:**

1. A clinician collects cheap data: age, body measurements, blood pressure, and a few lifestyle questions
2. The model predicts risk for each of the 4 conditions
3. High-risk patients are referred for confirmatory blood tests

---

## The 4 conditions predicted

| Condition | Best Model | Test AUC | Train–Test Gap |
|---|---|---|---|
| Diabetes | XGBoost | 0.819 | 0.042 |
| Cardiovascular disease (CVD) | XGBoost | 0.810 | 0.028 |
| Hypertension | XGBoost | 0.796 | 0.022 |
| Metabolic syndrome | Random Forest | 0.802 | 0.023 |

All gaps are below 0.05, indicating good fit with no overfitting.

---

## Why this project was rebuilt (the data-leakage story)

An earlier version of this project reported ~99% accuracy. That number was **not real** — it was caused by **data leakage**: columns used to *define* a disease were also being fed in as *features* to predict that same disease. The model was effectively being shown the answer.

This version was rebuilt from scratch with strict leakage prevention. The honest result (~0.80 AUC) is far more valuable than the fake 99%, because it reflects genuine predictive ability.

### The core rule followed

> Any column used to **create** a disease label must **not** be used as a **feature** to predict that same disease.

### How leakage was prevented

- **Label-maker columns dropped per model.** For example, HbA1c and glucose build the diabetes label, so they are removed from the diabetes feature set.
- **All diagnosis columns removed globally.** A known diagnosis (e.g. "doctor told you that you have diabetes") leaks risk information into every model, so all diagnosis columns are excluded from every model.
- **Expensive lab values excluded as inputs.** To make this a true screening tool, all blood-test results (cholesterol panel, CRP, kidney/liver markers, HbA1c, glucose, insulin) are treated as *outcomes to test for later*, not as model inputs.
- **Body-size leakage handled carefully.** For body-size-defined targets, correlated measurements (e.g. waist is ~0.93 correlated with BMI) are also removed to prevent indirect reconstruction of the label.

---

## Data

- **Source:** CDC NHANES 2021–2023 cycle
- **Files used:** 20 NHANES data files (demographics, examination, laboratory, questionnaire), joined on the participant ID `SEQN`
- **Population after cleaning:** 8,112 adults (aged 18+, non-pregnant)
- **Metabolic syndrome subset:** 3,534 fasting participants (the label requires fasting glucose and triglycerides to be valid)

### Cleaning steps

1. Kept only relevant columns from each of the 20 files
2. Joined all tables on `SEQN` with left joins, starting from demographics
3. Filtered to adults (age >= 18) and removed pregnant participants
4. Replaced NHANES special codes ("Refused" / "Don't know") with missing values, using the correct code for each column's scale (e.g. 7/9 for yes-no items, 777/999 for drink counts, 7777/9999 for activity minutes), while leaving genuine values untouched (e.g. 7 hours of sleep is real, not a code)
5. Built the 4 disease labels from diagnosis answers and medical thresholds
6. Engineered clean features, then dropped the raw columns they replaced

---

## Features used (cheap inputs only)

**Demographics:** age, gender, race/ethnicity, income-to-poverty ratio, education

**Body measurements:** weight, height, waist, BMI

**Blood pressure (averaged over 3 readings):** systolic, diastolic, pulse

**Lifestyle:** smoking status, alcohol intake, physical activity, sleep hours, depression score (PHQ-9)

> Each model drops the specific inputs that would leak its own label. For example, hypertension does not use blood-pressure readings (they define the label), and metabolic syndrome does not use waist.

---

## How the labels were defined

- **Diabetes:** doctor diagnosis OR HbA1c >= 6.5%
- **CVD:** self-reported heart attack OR coronary heart disease
- **Hypertension:** doctor diagnosis of high blood pressure
- **Metabolic syndrome:** 3 or more of 5 standard criteria (waist, triglycerides, HDL, blood pressure, glucose), computed on fasting participants

---

## Modelling approach

- **Algorithms compared:** Logistic Regression, Random Forest, XGBoost
- **Validation:** 80/20 train–test split, plus 5-fold stratified cross-validation
- **Imbalance handling:** balanced class weights (all conditions are minority classes)
- **Overfitting control:** regularized tree depth, minimum leaf sizes, learning-rate and penalty tuning; rare conditions (e.g. CVD) use tighter settings
- **Model selection:** for each condition, the algorithm with the highest test AUC and an acceptable train–test gap was chosen

### Feature importance (sanity check)

The most influential features for each condition match known medical risk factors, confirming the models learned real patterns rather than noise:

- **Diabetes:** age, physical activity, waist, BMI
- **CVD:** age, physical activity, smoking
- **Hypertension:** age, BMI, physical activity
- **Metabolic syndrome:** BMI, weight, age

---

## Project structure

```
nhanes_risk_prediction/
├── data/
│   └── raw/                   # 20 NHANES .xlsx files
├── models/                    # saved trained models
│   ├── diabetes_model.pkl     + diabetes_features.pkl
│   ├── cvd_model.pkl          + cvd_features.pkl
│   ├── hypertension_model.pkl + hypertension_features.pkl
│   └── metabolic_model.pkl    + metabolic_features.pkl
├── nhanes_model.ipynb         # full pipeline: load -> clean -> label -> train
└── README.md
```

---

## Key takeaways

- Honest, leakage-free models are more trustworthy than artificially perfect ones
- The same column can be a safe feature for one model and a leak for another — leakage must be handled per model
- A screening tool built on cheap inputs has real-world value: it directs limited testing resources to the patients who need them most

---

## Limitations

- Models are trained on a single NHANES cycle (2021–2023); performance on other populations is not guaranteed
- Self-reported diagnoses and questionnaire data carry inherent reporting bias
- This tool supports screening and is not a substitute for clinical diagnosis or professional medical judgment

---

## Acknowledgements

Data from the U.S. CDC National Health and Nutrition Examination Survey (NHANES), 2021–2023 cycle. For educational use.
