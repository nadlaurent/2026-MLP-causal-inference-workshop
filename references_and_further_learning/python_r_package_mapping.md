# Python-to-R Package Mapping for Causal Inference

Maps core analytical packages used in the SIOP 2026 Causal Inference Master Tutorial to their R equivalents. Use when porting analyses to R or collaborating with R users.

**Source:** `supp_functions/causal_diagnostics.py`, `supp_functions/causal_inference_modelling.py`

---

## Core Analytical Packages

| Python | R | Purpose |
|--------|---|---------|
| **statsmodels** GEE | `geepack::geeglm` | IPTW outcome models (clustered data) |
| **statsmodels** GLM | `stats::glm` | Propensity score models |
| **lifelines** | `survival` | Cox PH, Kaplan-Meier, log-rank |
| **econml** DML | `DoubleML` | ATE, ATT, CATE (linear DML) |
| **econml** CausalForestDML | `grf::causal_forest` | Heterogeneous treatment effects |
| Custom IPTW | `WeightIt::weightit` | Propensity weights (ATE/ATT) |
| Custom E-value | `EValue::evalue` | Sensitivity to unmeasured confounding |

---

## Quick Reference

| Python | R |
|--------|---|
| `smf.gee(..., groups=cluster)` | `geeglm(..., id = cluster)` |
| `smf.glm(..., family=Binomial())` | `glm(..., family = binomial)` |
| `CoxPHFitter` | `coxph(Surv(...) ~ ...)` |
| `KaplanMeierFitter` | `survfit(Surv(...) ~ ...)` |
| `DML` (econml) | `DoubleMLPLR` / `DoubleMLIRM` |
| `CausalForestDML` | `grf::causal_forest` |
| Manual propensity + weights | `weightit(treatment ~ ., method = "glm", estimand = "ATE")` |
| `multipletests(..., method='fdr_bh')` | `p.adjust(..., method = "BH")` |

---

## Supporting Packages

| Python | R |
|--------|---|
| `numpy`, `pandas` | base R, `tibble`, `data.table` |
| `scipy.stats.chi2_contingency` | `chisq.test` |
| `statsmodels` VIF | `car::vif` |
| `sklearn` (RF, AUC, CV) | `randomForest`/`ranger`, `pROC`, `caret` |
| `matplotlib` | `ggplot2` |
| `openpyxl` (via pandas) | `writexl`, `openxlsx` |

---

## Version Notes

- **Python:** See [requirements.txt](../requirements.txt) (statsmodels ≥0.14.2, econml ≥0.15.0, lifelines ≥0.27.0).
- **R:** DoubleML, grf, geepack, EValue, WeightIt available on CRAN.
