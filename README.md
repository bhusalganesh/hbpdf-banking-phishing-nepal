# hbpdf-banking-phishing-nepal
# HBPDF: Hybrid Banking Phishing Defence Framework — Layer 2 Validation

This repository contains the reproducible machine-learning experiment for the
Layer 2 (ML-based real-time transaction detection) component of the HBPDF
framework described in the paper:

> *A Hybrid Detection and Prevention Framework for Banking Phishing Scams in
> Emerging Digital Economies: Evidence from Nepal.*

The detection layer is validated on the public **PaySim** mobile-money dataset,
whose transfer-and-cash-out fraud topology mirrors the mule-account / rapid
cash-out patterns documented by the Financial Intelligence Unit of Nepal
Rastra Bank (FIU-Nepal, 2024).

## Results (real run, fixed seed = 42)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|-----------|--------|------|---------|--------|
| Isolation Forest | 0.318 | 0.466 | 0.378 | 0.885 | 0.264 |
| Random Forest | 0.998 | 0.997 | 0.997 | 0.999 | 0.998 |
| Gradient Boosting | 0.999 | 0.997 | 0.998 | 0.998 | 0.991 |
| **Hybrid (proposed)** | **0.999** | **0.997** | **0.998** | **0.999** | **0.995** |

3-fold cross-validation ROC-AUC = 0.998 (std 0.0008). Full output in
`paysim_results.json`.

## How to reproduce

1. Download the PaySim dataset (free) from Kaggle:
   https://www.kaggle.com/datasets/ealaxi/paysim1
2. Unzip and rename the CSV to `paysim.csv`, placing it in this folder.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the experiment:
   ```
   python paysim_experiment_fast.py --data paysim.csv
   ```
   (`paysim_experiment_fast.py` subsamples the legitimate-class majority for
   tractable training on commodity hardware while keeping every fraud case;
   `paysim_experiment.py` runs on the full dataset if you have sufficient RAM.)

The script prints the results table and writes `paysim_results.json`.

## Method summary

- Evaluation is restricted to TRANSFER and CASH_OUT transactions, the only
  PaySim types in which fraud occurs.
- FIU-Nepal qualitative red-flag indicators are translated into engineered
  features (balance-emptying, amount-to-balance ratio, balance-discrepancy
  errors, transfer/cash-out channel; see paper Table III).
- The proposed **Hybrid** model is a feature-augmented Gradient Boosting
  classifier in which an Isolation Forest anomaly score is injected as an
  additional input feature — structurally aligned with production
  fraud-detection systems.

## Note on scores

PaySim fraud is defined by deterministic balance-accounting signatures, so
supervised models achieve near-perfect separation; the ~0.99 range is
consistent with the published PaySim literature. Operational bank data, with
noisier labels and adversarial adaptation, would yield lower and more realistic
scores. PaySim is used here as a transparent, reproducible, topologically
representative proxy, not as a measurement of field performance in Nepal.

## Citation

If you use this code, please cite the paper (details to be added on
publication) and the PaySim dataset:

> E. A. Lopez-Rojas, A. Elmir, and S. Axelsson, "PaySim: A financial mobile
> money simulator for fraud detection," in *Proc. 28th European Modeling and
> Simulation Symposium (EMSS)*, 2016, pp. 249–255.

## License

MIT — see `LICENSE`.
