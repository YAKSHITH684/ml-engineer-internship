\# Nested Cross-Validation and Statistical Validation



\## Experimental setup

\- Training data: 80% stratified training split.

\- Outer validation: 5-fold StratifiedKFold.

\- Inner hyperparameter tuning: 3-fold StratifiedKFold.

\- XGBoost candidates: 10 per outer fold.

\- Total XGBoost candidate fits: 150.

\- Both models evaluated on identical outer validation folds using PR-AUC.



\## Results



| Metric | Logistic Regression | Nested XGBoost |

|---|---:|---:|

| Mean PR-AUC | 0.223106 | 0.248868 |

| Fold standard deviation | 0.032277 | 0.038685 |



XGBoost achieved a mean PR-AUC improvement of 0.025762 (approximately 11.55% relative to Logistic Regression). It outperformed the baseline on all five outer folds.



\## Statistical analysis

\- Paired t-test p-value: 0.029833.

\- Wilcoxon signed-rank p-value: 0.062500.

\- Exploratory 95% confidence interval: 0.004111 to 0.047413.



The evidence is encouraging but not conclusive. Only five outer folds were evaluated, and their training sets overlap. Statistical tests and confidence intervals should be interpreted cautiously.



\## Gate 2 assessment



The tuned model improved over the baseline on all five outer folds. However, nested XGBoost fold standard deviation was 0.038685, above the 0.005 absolute PR-AUC threshold.



The statistical evidence is mixed: the paired t-test is below 0.05, while the Wilcoxon test is above 0.05.



Gate 2 is not declared fully passed. Confirm the 0.5% variability requirement with the internship mentor. Repeated nested cross-validation may provide further evidence about model stability.

