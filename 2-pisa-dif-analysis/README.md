# PISA DIF Analysis (US Sample)

## Overview

This project detects **Differential Item Functioning (DIF)** in PISA 2022 reading items using the US sample. Multiple DIF detection methods are applied across 12 grouping variables, and Rosenbaum sensitivity analysis is used to assess the robustness of flagged items.

### DIF Methods

| Method | Function |
|--------|----------|
| Logistic Regression DIF | `difR::difLogistic` |
| Benjamini–Hochberg correction | `p.adjust(..., method = "BH")` |
| Mantel–Haenszel | `difR::difMH` |
| Wald Test (IRT-based) | `difR::difRaju` |
| Rosenbaum Sensitivity Analysis | `rbounds::binarysens` |

### Grouping Variables

| Variable | Description |
|----------|-------------|
| Gender | Student gender |
| Books at Home | Proxy for home literacy environment |
| HISCED | Highest parental education |
| IMMIG | Immigration status |
| BELONG | Sense of belonging at school |
| BULLIED | Experience of bullying |
| ESCS | Socioeconomic, cultural, and social status index |
| Math Self-Efficacy | Student math self-efficacy |
| Math Preference | Enjoyment of mathematics |
| Math Ease/Anxiety | Math anxiety scale |
| REPEAT | Grade repetition |
| Single Parent | Single parent household |

> Note: Language background was excluded due to numerical instability.

---

## Key Findings

- Logistic regression detects DIF broadly across grouping variables.
- BH correction substantially reduces the number of flagged items.
- MH and Wald tests rarely confirm logistic findings.
- Most DIF is fragile under Rosenbaum sensitivity analysis (low critical λ).
- **ESCS** and **REPEAT** show the strongest evidence of robust DIF.

---

## Data

Data is from the **PISA 2022 International Database** (US sample).

- Download: [https://www.oecd.org/pisa/data/](https://www.oecd.org/pisa/data/)
- Required file: Student questionnaire + cognitive data (SAS or SPSS format)
- Load into R as `pisa_us` (filtered to US students)

---

## Reproducing the Analysis

```r
# Step 1: Install dependencies
install.packages(c("difR", "rbounds", "dplyr", "optmatch", "mirt"))

# Step 2: Load the PISA US data (not included — see Data section above)
# pisa_us <- <your loading code here>

# Step 3: Run analysis
source("analysis.R")
```

---

## Files

| File | Description |
|------|-------------|
| `analysis.R` | Full analysis pipeline: data prep → DIF methods → BH correction → sensitivity |

---

## Replication Instructions

1. Load cleaned PISA 2022 US dataset as `pisa_us`
2. Run preprocessing steps (identify dichotomous items, recode, clean)
3. Define each grouping variable as `group_var` (0/1 binary)
4. Run DIF methods per grouping variable
5. Apply BH correction to logistic p-values
6. Run Rosenbaum sensitivity analysis on flagged items
7. Aggregate results into summary tables

---

## Dependencies

```r
library(difR)
library(rbounds)
library(dplyr)
library(mirt)
library(optmatch)
```

---

## Author

Luke Lin — University of Michigan
