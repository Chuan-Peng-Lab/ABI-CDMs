# Data sources and analysis samples

This release contains nine trial-level CSV files. Eight studies contribute to the cross-sectional analysis, and `clayson2024.csv` is analyzed separately as a test–retest dataset. `erb2023.csv` is excluded from the released analysis and is not distributed in ABI-CDMs.

## Raw-file inventory

`Raw N` is the number of unique participant identifiers present in each distributed CSV before the project preprocessing rules are applied.

| File | Study code | Tasks represented | Raw N | Analysis role |
|---|---:|---|---:|---|
| `clayson2024.csv` | C24 | Flanker variants | 191 | Test–retest only |
| `clayson2025.csv` | C25 | Flanker, Stroop | 169 | Cross-sectional |
| `eisenberg2019.csv` | E19 | Flanker, Simon, Stroop | 523 | Cross-sectional |
| `hedge2018.csv` | H18 | Flanker, Simon, Stroop | 131 | Cross-sectional |
| `kucina2023.csv` | K23 | Flanker, Simon, Stroop | 181 | Cross-sectional |
| `lee2025.csv` | L25 | Flanker, Stroop | 7 | Cross-sectional; first session only |
| `reymermet2018.csv` | R18 | Flanker, Simon, Stroop | 262 | Cross-sectional |
| `ulrich2015.csv` | U15 | Flanker, Simon | 16 | Cross-sectional |
| `whitehead2019.csv` | W19 | Flanker, Simon, Stroop | 178 | Cross-sectional |

## Final analysis samples

The manuscript reports 1,375 participants across the eight cross-sectional studies. Final task-level N values differ from raw N because preprocessing selects eligible tasks and sessions, removes invalid trials, and applies the project quality criteria.

| Study | Final analysis N by task |
|---|---|
| C25 | Flanker 159; Stroop 159 |
| E19 | Flanker 504; Simon 504; Stroop 504 |
| H18 | Flanker 53; Simon 102; Stroop 53 |
| K23 | Flanker 60; Simon 30; Stroop 30 |
| L25 | Flanker 6; Stroop 6 |
| R18 | Flanker 237; Simon 237; Stroop 237 |
| U15 | Flanker 16; Simon 16 |
| W19 | Flanker 178; Simon 178; Stroop 178 |

C24 is not counted in that cross-sectional total. Its four flanker variants/sessions are used for temporal-stability analyses; the final retest analysis uses 150 eligible participants.

## Preprocessing contract

The canonical entry point is `scripts/01_preprocessing/prepare_datasets.py`. It reads only the nine files listed above and writes:

- `results/intermediate/datasets_cross_sectional.h5`;
- `results/intermediate/datasets_retest.h5`.

Downstream scripts must consume these generated stores rather than read raw CSVs directly.

## Dataset citations

- **Clayson et al. (2024)** — test–retest flanker data (C24).
- **Clayson et al. (2025)** — cross-task, multi-site conflict-task data (C25).
- **Eisenberg et al. (2019)** — self-regulation task battery (E19).
- **Hedge, Powell, and Sumner (2018)** — reliability-paradox dataset (H18).
- **Kucina et al. (2023)** — individual-differences conflict-task data (K23).
- **Lee et al. (2025)** — repeated conflict-task assessments (L25).
- **Rey-Mermet, Gade, and Oberauer (2018)** — inhibition and Stroop data (R18).
- **Ulrich et al. (2015)** — conflict-task processing data (U15).
- **Whitehead, Brewer, and Blais (2019)** — cognitive-control reliability data (W19).

Use the accompanying manuscript reference list for complete bibliographic records. This file documents release scope and analysis provenance; it does not replace the original dataset licenses or publications.
