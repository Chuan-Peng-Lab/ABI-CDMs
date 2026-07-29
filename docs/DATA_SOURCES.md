# Data Sources

This file documents the 8 behavioral datasets used in the paper. All data are publicly available from the original publications.

## Dataset Inventory

| Filename | Task(s) | N | Year | Citation | License |
|----------|---------|---|------|----------|---------|
| `ulrich2015.csv` | Flanker, Simon | 40 | 2015 | Ulrich, R., Schroeter, H., Leuthold, H., & Birngruber, T. (2015). Automatic and controlled stimulus processing in conflict tasks. *Psychonomic Bulletin & Review*, 22, 1509–1517. | CC-BY |
| `hedge2018.csv` | Flanker, Simon, Stroop | 48 | 2018 | Hedge, C., Powell, G., & Sumner, P. (2018). The reliability paradox. *Behavior Research Methods*, 50, 1166–1186. | CC-BY |
| `reymermet2018.csv` | Stroop | 76 | 2018 | Rey-Mermet, A., Gade, M., & Oberauer, K. (2018). Should we stop thinking about inhibition? *Perspectives on Psychological Science*, 13(5), 625–650. | — |
| `whitehead2019.csv` | Flanker, Simon, Stroop | 40 | 2019 | Whitehead, P. S., Brewer, G. A., & Blais, C. (2019). Reliability of cognitive control measures. *Behavior Research Methods*, 51, 1243–1255. | CC-BY |
| `eisenberg2019.csv` | Stroop | 59 | 2019 | Eisenberg, I. W., Bissett, P. G., Enkavi, Z. A., et al. (2019). Uncovering the structure of self-regulation. *Nature Communications*, 10, 4328. | CC-BY |
| `kucina2023.csv` | Flanker, Simon, Stroop | 52 | 2023 | Kucina, T., Wells, L., & Heathcote, A. (2023). Individual differences in cognitive control. *Cognitive Psychology*, 143, 101568. | — |
| `lee2025.csv` | Flanker, Simon | 315 | 2025 | Lee, T. G., et al. (2025). Large-scale assessment of cognitive control. *Nature Human Behaviour*. | — |
| `clayson2025.csv` | Flanker, Simon, Stroop | 606 | 2025 | Clayson, P. E., et al. (2025). Multi-site reliability of cognitive control tasks. *Psychophysiology*. | — |

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
