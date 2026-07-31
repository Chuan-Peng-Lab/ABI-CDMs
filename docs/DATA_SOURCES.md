# Data Sources

This file documents the behavioral datasets used in the paper. All data are publicly available from the original publications.

## Dataset Inventory

| Filename | Task(s) | N | Mean trials* | Cross-task design | No. of sessions | Stimulus | Design Note | Year | Citation | License |
|----------|---------|---|--------------|-------------------|-----------------|----------|-------------|------|----------|---------|
| `ulrich2015.csv` | Flanker, Simon | 40 | — | — | — | — | — | 2015 | Ulrich, R., Schroeter, H., Leuthold, H., & Birngruber, T. (2015). Automatic and controlled stimulus processing in conflict tasks. *Psychonomic Bulletin & Review*, 22, 1509–1517. | CC-BY |
| `hedge2018.csv` | Flanker, Simon, Stroop | 48 | — | — | — | — | — | 2018 | Hedge, C., Powell, G., & Sumner, P. (2018). The reliability paradox. *Behavior Research Methods*, 50, 1166–1186. | CC-BY |
| `reymermet2018.csv` | Stroop | 76 | — | — | — | — | — | 2018 | Rey-Mermet, A., Gade, M., & Oberauer, K. (2018). Should we stop thinking about inhibition? *Perspectives on Psychological Science*, 13(5), 625–650. | — |
| `whitehead2019.csv` | Flanker, Simon, Stroop | 40 | — | — | — | — | — | 2019 | Whitehead, P. S., Brewer, G. A., & Blais, C. (2019). Reliability of cognitive control measures. *Behavior Research Methods*, 51, 1243–1255. | CC-BY |
| `eisenberg2019.csv` | Stroop | 59 | — | — | — | — | — | 2019 | Eisenberg, I. W., Bissett, P. G., Enkavi, Z. A., et al. (2019). Uncovering the structure of self-regulation. *Nature Communications*, 10, 4328. | CC-BY |
| `kucina2023.csv` | Flanker, Simon, Stroop | 52 | — | — | — | — | — | 2023 | Kucina, T., Wells, L., & Heathcote, A. (2023). Individual differences in cognitive control. *Cognitive Psychology*, 143, 101568. | — |
| `lee2025.csv` | Flanker, Simon | 315 | — | — | — | — | — | 2025 | Lee, T. G., et al. (2025). Large-scale assessment of cognitive control. *Nature Human Behaviour*. | — |
| `clayson2024.csv` | Flanker | 150 | 464 | No | 4 | 5 arrows | Compared 3 versions of the flanker task; 24 practice trials. Task A: 900 trials, 45% congruent / 55% incongruent, 1600 ms response window. Task B: 330 trials, 50% congruent / 50% incongruent, 2000 ms response window, block-wise feedback. Task C: 400 trials, 50% congruent / 50% incongruent, response window equals ITI duration, block-wise feedback. | 2024 | Clayson, P. E., et al. (2024). Reliability of conflict task performance. *Psychophysiology*. | — |
| `clayson2025.csv` | Flanker, Simon, Stroop | 606 | — | Yes | 1 | 5 arrows / color words | Multi-site reliability study. Flanker & Stroop response windows limited to 100–700 ms; Stroop includes neutral words and 3 congruency conditions (1/3 each); 20 practice trials. | 2025 | Clayson, P. E., et al. (2025). Multi-site reliability of cognitive control tasks. *Psychophysiology*. | — |

\* Mean trials per subject per task; "—" indicates not recorded in the original source.

### Notes on Clayson datasets

- **C24 (Clayson et al., 2024)**: In this project only the **Flanker** task is used. The three flanker versions (ffa, ffb, ffc) plus the standard flanker (flk) are treated as **four sessions** for retest/reliability analysis. Subjects must contribute data to more than one session to be included.
- **C25 (Clayson et al., 2025)**: A cross-task design with Flanker, Simon, and Stroop collected within a single session; used for multi-site reliability.

## Data Format

Each CSV file contains trial-level data with the following columns (may vary by dataset):

- `subject_id`: Participant identifier
- `task_name`: Task type (flanker, simon, stroop)
- `congruency`: Trial condition (congruent, incongruent, neutral)
- `rt`: Reaction time in milliseconds
- `accuracy`: Response accuracy (1 = correct, 0 = error)
- `session_id`: Session number (for retest datasets)

## Usage

These data files are used as input to the `scripts/01_preprocessing/21datasets_preprocessing.py` script, which standardizes the format across datasets before model fitting.
