\# Week 2 Model Report — Classical ML and Anti-Fraud Pipeline



\*\*Project:\*\* Attenomy ML Engineering Internship  

\*\*Week:\*\* 2 — Classical ML, Feature Engineering and Model Interpretability  

\*\*Report date:\*\* 30 October 2026  

\*\*Status:\*\* Analysis completed; additional validation required before production deployment.



\---



\## 1. Executive Summary



This week focused on developing and evaluating a machine learning pipeline for telemetry anomaly detection. The work included baseline model evaluation, feature engineering, hyperparameter tuning, model comparison, SHAP interpretability, error analysis, subgroup evaluation and probability calibration.



The tuned XGBoost model achieved a best cross-validation PR-AUC of \*\*0.24666\*\* during randomized hyperparameter search. The selected operating threshold was approximately \*\*0.788394\*\*, chosen using out-of-fold predictions to optimize F1-score.



The analysis identified an important relationship between error-code availability and anomaly predictions. It also revealed performance differences across interaction types, channels and regions.



\*\*Recommendation:\*\* Retain the saved XGBoost pipeline as the primary experimental model. Treat the isotonic-calibrated version as a candidate requiring further validation. Do not deploy automatic blocking decisions until calibration, threshold selection and operational costs are validated.



\## 2. Data and Experimental Methodology



The experiments used the Attenomy telemetry dataset with a stratified training/testing split.



\- Training records: 20,000

\- Testing records: 5,000

\- Test anomaly prevalence: 3.92%

\- Evaluation approach: Stratified 5-fold cross-validation and held-out testing

\- Primary evaluation metric: PR-AUC



Because anomalies are relatively rare, accuracy alone is not sufficient. Precision, recall, F1-score, ROC-AUC and PR-AUC were considered.



The preprocessing and feature engineering steps were integrated into the model pipeline to reduce the risk of data leakage. Training-derived mappings were used when transforming validation and test records.



\## 3. Baseline Models and Feature Engineering



\### 3.1 Baseline evaluation



Five baseline approaches were evaluated:



\- Dummy Classifier

\- Logistic Regression

\- K-Nearest Neighbors

\- Decision Tree

\- Random Forest



The baseline cross-validation PR-AUC results included:



| Model | PR-AUC |

|---|---:|

| Logistic Regression | 0.2231 |

| Decision Tree | 0.1984 |

| Random Forest | 0.1960 |

| K-Nearest Neighbors | 0.0846 |

| Dummy Classifier | 0.0392 |



Logistic Regression provided the strongest baseline PR-AUC among these models.



\### 3.2 Engineered features



Eight features were implemented:



1\. `payload\_per\_second`

2\. `retry\_latency\_interaction`

3\. `channel\_frequency`

4\. `region\_frequency`

5\. `hour\_sin`

6\. `hour\_cos`

7\. `account\_mean\_latency`

8\. `latency\_bin`



These features represent payload throughput, retry behaviour, categorical frequency, cyclical time patterns and account-level latency behaviour.



The feature engineering process was integrated into the pipeline with training-derived statistics and mappings.



\### 3.3 Ablation study



Seven feature configurations were evaluated using 5-fold cross-validation.



| Configuration | Mean PR-AUC | CV standard deviation |

|---|---:|---:|

| Without cyclical time | 0.212028 | 0.011142 |

| All engineered features | 0.210657 | 0.010773 |

| Original baseline features | 0.210752 | 0.010514 |



The engineered features did not demonstrate a clear PR-AUC improvement over the original baseline configuration in this ablation experiment.



\*\*Engineering conclusion:\*\* Feature engineering should be retained only when it provides measurable value. Further investigation should focus on feature quality, missingness and interaction-specific behaviour rather than assuming that adding features improves performance.



\## 4. Hyperparameter Tuning and Model Comparison



\### 4.1 XGBoost tuning



RandomizedSearchCV completed 50 parameter combinations.



Best cross-validation PR-AUC:



\*\*0.24666\*\*



Selected parameters:



\- `n\_estimators`: 100

\- `max\_depth`: 3

\- `learning\_rate`: 0.03

\- `subsample`: 0.9

\- `colsample\_bytree`: 0.8

\- `min\_child\_weight`: 3

\- `gamma`: 0.5

\- `reg\_lambda`: 5



The selected model was saved as:



`models/best\_model.joblib`



\### 4.2 XGBoost versus LightGBM



The 5-fold comparison produced:



| Metric | XGBoost | LightGBM |

|---|---:|---:|

| PR-AUC | 0.2515 | 0.2407 |

| ROC-AUC | 0.8035 | 0.8054 |

| F1-score | 0.0221 | 0.2536 |

| Recall | 0.0112 | 0.7436 |

| Precision | 0.6600 | 0.1529 |



The comparison used different class-weighting configurations: LightGBM used balanced class weights, while XGBoost did not use the same weighting configuration. Therefore, the threshold-dependent metrics should not be interpreted as a perfectly controlled comparison.



The results suggest that both models require further evaluation under consistent class weighting and threshold-selection procedures.



\### 4.3 Operating threshold



The selected XGBoost threshold was approximately \*\*0.788394\*\*, based on out-of-fold training predictions.



Out-of-fold results at the selected threshold:



\- Precision: 0.2457

\- Recall: 0.4559

\- F1-score: 0.3193



On the held-out test set, the error analysis identified:



\- True positives: 80

\- True negatives: 4,518

\- False positives: 286

\- False negatives: 116



This corresponds to approximately:



\- Accuracy: 91.96%

\- Precision: 21.86%

\- Recall: 40.82%

\- F1-score: 28.47%



The threshold is an experimental operating point, not a universal production threshold. The final threshold should depend on the relative operational costs of false positives and false negatives.



\## 5. Model Interpretability Using SHAP



SHAP was used to understand the features influencing XGBoost predictions.



The following figures were generated:



\- `reports/figures/shap\_summary.png`

\- `reports/figures/shap\_importance.png`

\- `reports/figures/shap\_waterfall\_1.png`

\- `reports/figures/shap\_waterfall\_2.png`

\- `reports/figures/shap\_waterfall\_3.png`



The analysis used 1,000 sampled test records and a background sample of 100 transformed test records. The transformed feature matrix contained 39 features.



\### 5.1 Global feature importance



The leading features by mean absolute SHAP value were:



| Feature | Mean absolute SHAP |

|---|---:|

| error\_code\_MISSING | 0.402662 |

| retry\_count | 0.184975 |

| payload\_size\_kb | 0.162062 |

| retry\_latency\_interaction | 0.045870 |

| payload\_per\_second | 0.038574 |

| session\_duration\_sec | 0.034390 |

| latency\_ms | 0.033701 |

| latency\_bin | 0.022162 |



The missing error-code indicator had the largest mean absolute SHAP value, followed by retry count and payload size.



Three individual waterfall plots were generated to explain individual model predictions.



\*\*Interpretation limitation:\*\* These SHAP values represent contributions to the model's raw output margin, not direct changes in predicted probability. Mean absolute SHAP measures contribution magnitude, not whether a feature increases or decreases anomaly risk. Direction should be interpreted using the beeswarm and individual waterfall plots.



\## 6. Error Analysis



The error analysis inspected 20 high-confidence misclassifications:



\- 10 false positives

\- 10 false negatives



The selected examples were saved in:



`reports/error\_analysis\_20.csv`



The analysis used the selected out-of-fold threshold of approximately 0.788394.



\### 6.1 Observed patterns



\- All 10 selected false positives had an explicit error code.

\- All 10 selected false negatives had a missing error code.

\- The selected false positives generally had retry counts of zero or one.

\- The selected false negatives had retry counts of zero.

\- False negatives generally showed longer session durations.



The full test-set analysis found:



| Prediction group | Error code present | Error code missing |

|---|---:|---:|

| Correct predictions | 2.6% | 97.4% |

| False negatives | 4.3% | 95.7% |

| False positives | 96.5% | 3.5% |



The observed anomaly rate was approximately 2.46% when the error code was missing and 20.65% when it was present.



This indicates that error-code availability is strongly associated with the model's predictions and the observed anomaly labels.



\### 6.2 Engineering interpretation



The model may rely heavily on explicit error-code availability. This can contribute to false positives when normal records contain error codes and false negatives when anomalous records do not.



However, this association does not establish causation. The telemetry logging process and the meaning of missing error codes must be investigated before changing the feature.



Negative session-duration values were also observed in the data. These values should be investigated as possible data-quality or timestamp-processing issues.



\## 7. Subgroup Performance Audit



A subgroup performance audit was performed across:



\- Channel

\- Region

\- Client tier

\- Interaction type



The detailed results are saved in:



`reports/subgroup\_fairness\_audit.csv`



\### 7.1 Key observations



\- Python SDK showed lower recall than the other evaluated channels.

\- The ap-south-1 region showed the lowest regional recall among the evaluated regions.

\- Model inference showed the weakest recall and PR-AUC among interaction types.

\- Agent orchestration also showed relatively low recall.

\- Batch sync and API query had stronger PR-AUC results than model inference.



The model-inference subgroup had PR-AUC of approximately 0.1264 and recall of 0.2800.



Some groups had small sample sizes. VIP had nine actual anomalies, while the missing client-tier group had eight. These results should not be treated as reliable evidence of systematic group differences.



\*\*Fairness limitation:\*\* The available audit covers operational subgroups, not demographic groups. Differences in subgroup performance do not by themselves prove unfairness or identify their cause. Further investigation should include sample-size-aware confidence intervals and additional representative data.



\## 8. Probability Calibration



Calibration was evaluated using out-of-fold training predictions and held-out test predictions.



The following figure was generated:



`reports/figures/calibration\_curve.png`



\### 8.1 Calibration results



| Metric | Original model | Isotonic calibrated |

|---|---:|---:|

| Brier score | 0.146420 | 0.033131 |

| Log loss | 0.477739 | 0.132153 |

| PR-AUC | 0.243773 | 0.207375 |



The test anomaly prevalence was 3.92%.



The original model's mean predicted probability was approximately 34.77%, substantially higher than the observed anomaly prevalence. The isotonic-calibrated model's mean predicted probability was approximately 3.96%.



Isotonic calibration substantially improved Brier score and log loss, but PR-AUC decreased. This indicates a trade-off between probability calibration and ranking performance.



\### 8.2 Calibration limitation and recommendation



The isotonic calibrator was fitted on out-of-fold training predictions and applied to predictions from the model fitted on the full training split. This can introduce a score-distribution mismatch.



Therefore:



\- Keep the original saved XGBoost pipeline as the primary experimental artifact.

\- Retain isotonic calibration as a separate candidate.

\- Validate calibration with a strictly cross-fitted or nested calibration procedure.

\- Select a new operating threshold for calibrated probabilities using training-only validation.

\- Do not reuse the original 0.788394 threshold on calibrated probabilities.



The calibration results are promising but are not sufficient evidence for production deployment.



\## 9. Engineering Hypotheses and Recommendations



The experiments suggest the following evidence-backed hypotheses for further investigation.



\### Hypothesis 1: Error-code logging behaviour affects anomaly detection



The error-code missingness indicator dominates global SHAP importance, and the anomaly rate is substantially higher when an error code is present.



\*\*Recommendation:\*\* Investigate upstream logging semantics, missing-value generation and whether error codes are consistently recorded across telemetry sources.



\### Hypothesis 2: Model performance depends on interaction type



Model inference and agent orchestration show weaker recall than some other interaction types.



\*\*Recommendation:\*\* Investigate workflow-specific telemetry features, collect additional examples and evaluate whether interaction-specific feature engineering improves recall.



\### Hypothesis 3: Telemetry quality issues may affect predictions



Negative session-duration values and missing telemetry fields indicate potential data-quality issues.



\*\*Recommendation:\*\* Add validation rules for timestamps, session duration, missing fields and invalid values before model inference.



\### Hypothesis 4: The decision threshold should reflect operational costs



The selected threshold balances precision and recall using F1-score, but operational requirements may prioritize a different trade-off.



\*\*Recommendation:\*\* Define acceptable false-positive and false-negative costs with the product team. Evaluate alternative thresholds before enabling automated actions.



\## 10. Week 3 Product Transition Plan



The next phase should focus on moving from offline experimentation to a controlled product-oriented inference workflow.



\### 10.1 Model serving



\- Build a FastAPI inference endpoint.

\- Load the saved model pipeline without separating preprocessing from prediction.

\- Validate incoming telemetry fields and data types.

\- Return an anomaly score, predicted class and model version.



\### 10.2 Explainability



\- Provide a human-readable explanation for high-risk predictions.

\- Use SHAP explanations to identify important contributing features.

\- Keep explanations separate from claims of causation.



\### 10.3 Monitoring



Monitor the following:



\- Prediction volume and anomaly rate.

\- Precision and recall when ground-truth labels become available.

\- Feature missingness and distribution drift.

\- Inference latency and API errors.

\- Performance across channels, regions and interaction types.

\- Calibration and threshold stability.



\### 10.4 Controlled deployment



Begin with shadow mode or human review rather than automatic blocking. Record model versions, thresholds, predictions and reviewer feedback. Only consider automated actions after sufficient validation and agreement on operational costs.



\## 11. Limitations and Final Assessment



The following limitations remain:



1\. Engineered features did not demonstrate a clear PR-AUC improvement in the ablation experiment.

2\. The XGBoost and LightGBM comparison used different class-weighting configurations.

3\. Cross-validation variability must be assessed against the internship's strict acceptance requirement.

4\. Statistical significance of the tuned model's improvement has not yet been established.

5\. SHAP values explain model behaviour but do not establish causal relationships.

6\. The subgroup audit is operational, not demographic.

7\. Isotonic calibration requires further cross-fitted validation.

8\. The selected threshold requires validation against product-level costs.



The internship acceptance criterion requires a CV standard deviation below 0.5% and evidence of statistically significant improvement. The current experiments do not yet establish that this requirement has been met. Therefore, Gate 2 should not be marked as passed without additional validation.



\## 12. Conclusion



Week 2 produced a complete experimental foundation for telemetry anomaly detection, including baseline evaluation, feature engineering, hyperparameter tuning, SHAP explanations, error analysis, subgroup evaluation and calibration experiments.



The tuned XGBoost model is a useful candidate for further evaluation. The most important engineering finding is the strong influence of error-code availability, which should be investigated at the data-collection level.



The next phase should prioritize validation, reliable probability estimates, threshold selection and a monitored inference API before any production automation.



\---



\*\*Key output files\*\*



\- Model: `models/best\_model.joblib`

\- Tuning results: `reports/tuning\_results.csv`

\- Ablation results: `reports/ablation\_results.csv`

\- Model comparison: `reports/model\_comparison.csv`

\- SHAP importance: `reports/shap\_feature\_importance.csv`

\- Error analysis: `reports/error\_analysis\_20.csv`

\- Subgroup audit: `reports/subgroup\_fairness\_audit.csv`

\- Calibration metrics: `reports/calibration\_metrics.json`

\- Figures: `reports/figures/`



