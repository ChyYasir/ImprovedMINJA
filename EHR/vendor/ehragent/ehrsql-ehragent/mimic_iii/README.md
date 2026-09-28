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

**Important gotcha, found 2026-09-28**: `mimic_iii.db` is tracked with Git LFS in the upstream
repo (`MINJA/.gitattributes`: `*.db filter=lfs ...`). If `MINJA/` was cloned without `git-lfs`
installed, `mimic_iii.db` will be a 134-byte LFS *pointer* text file, not the real ~99MB SQLite
database — `sqlite3.connect()` will open it without error, but any real query against it fails
with `Error: file is not a database`. `mimic_iii.sqlite` is a red herring here: it's a small
(~6KB) schema-only SQL text dump, not a usable database, and isn't LFS-tracked, so its small size
looks "normal" and doesn't tip you off that something's wrong. `SQLInterpreter` (in
`tools/tabtools.py`) specifically needs the real `mimic_iii.db`. To fix: install git-lfs
(`brew install git-lfs` on macOS), then inside `MINJA/` run `git lfs install --local && git lfs
pull` to materialize the real file, then copy it here. `file mimic_iii.db` should report
`SQLite 3.x database`, not `ASCII text`, when it's correct.
