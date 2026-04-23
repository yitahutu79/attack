# Gaussian Noise Robustness

Target FPR=0.05; threshold calibrated on clean benign windows. Eval seconds=238.27.

| Scenario | Sigma | F1 | Recall | Precision | Test FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 0.00 | 0.7552 | 0.6945 | 0.8276 | 0.1447 |
| Noise 1% | 0.01 | 0.7596 | 0.6967 | 0.8350 | 0.1377 |
| Noise 3% | 0.03 | 0.7710 | 0.7047 | 0.8512 | 0.1232 |
| Noise 5% | 0.05 | 0.7866 | 0.7248 | 0.8598 | 0.1182 |
