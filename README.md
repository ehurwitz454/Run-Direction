This pipeline predicts the direction of rushing plays in NFL games using player tracking data from the  2025 NFL Big Data Bowl dataset. It focuses on offensive line and tight end alignment/motion to model and interpret team and player tendencies.

├── main.py
├── data/
│   └── dataLoading.py
├── processing/
│   ├── merge.py
│   ├── pandasMerge.py
│   └── lineSet.py
├── analysis/
│   ├── Analysis.py
│   ├── beforeSnap.py
│   └── snapTuneOptuna.py
