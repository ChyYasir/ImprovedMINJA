# MIMIC-III data (not included in this repo)

This directory is required at runtime but its contents are intentionally excluded from version
control (clinical data).

`ImprovedMINJA/EHR/vendor/ehragent/tools/tabtools.py` hardcodes relative paths into this directory
(e.g. `../ehrsql-ehragent/mimic_iii/ADMISSIONS.csv`, `../ehrsql-ehragent/mimic_iii/mimic_iii.db`),
so the agent's generated code can query it directly, and
`ImprovedMINJA/EHR/harness/realistic_memory.py` (Phase 1) samples benign records from
`valid_filtered_20240104.json` here.

To populate this directory, copy the contents of `mimic_iii/` from either:

- `MINJA/EHR/ehragent/ehrsql-ehragent/mimic_iii/` (the vendor checkout this project builds on), or
- the upstream repo directly: [dsh3n77/MINJA](https://github.com/dsh3n77/MINJA),
  `EHR/ehragent/ehrsql-ehragent/mimic_iii/`

Expected files: `ADMISSIONS.csv`, `CHARTEVENTS.csv`, `COST.csv`, `DIAGNOSES_ICD.csv`,
`D_ICD_DIAGNOSES.csv`, `D_ICD_PROCEDURES.csv`, `D_ITEMS.csv`, `D_LABITEMS.csv`, `ICUSTAYS.csv`,
`INPUTEVENTS_CV.csv`, `LABEVENTS.csv`, `MICROBIOLOGYEVENTS.csv`, `OUTPUTEVENTS.csv`,
`PATIENTS.csv`, `PRESCRIPTIONS.csv`, `PROCEDURES_ICD.csv`, `TRANSFERS.csv`, `mimic_iii.db`,
`mimic_iii.sqlite`, `patient_only.json`, `train.json`, `valid.json`,
`valid_filtered_20240104.json`, `valid_filtered_20240104.jsonl`, `valid_preprocessed.json`,
`valid_preprocessed.jsonl` (~99MB total).
