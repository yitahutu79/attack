# SWaT Granularity Statistics

Window label rule: A window is labeled anomalous when the attack-record ratio within that window is >= Anomaly Ratio Threshold.

Records are counted after applying the same preprocessing as the main pipeline (drop non-feature columns, keep numeric features, drop NaN/inf rows).

| Granularity | Benign Records | Attack Records | Train Records | Calibration Records | Test Benign Records | Test Attack Records | Train Windows | Calibration Windows | Test Benign Windows | Test Anomalous Windows | Total Test Windows | Window Size | Stride | Anomaly Ratio Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1sec | 136800 | 90391 | 82080 | 27360 | 27360 | 90391 | 5123 | 1703 | 1703 | 5642 | 7345 | 128 | 16 | 0.15 |
| 3sec | 45600 | 29627 | 27360 | 9120 | 9120 | 29627 | 1703 | 563 | 563 | 1844 | 2407 | 128 | 16 | 0.15 |
| 5sec | 27360 | 17695 | 16416 | 5472 | 5472 | 17695 | 1019 | 335 | 335 | 1098 | 1433 | 128 | 16 | 0.15 |
| 10sec | 13680 | 16350 | 8208 | 2736 | 2736 | 16350 | 506 | 164 | 164 | 1014 | 1178 | 128 | 16 | 0.15 |

Auto-extracted archives:
- /Users/lijie/Desktop/work/attack/dataset/SWaT/extracted/attack_samples_10sec/attack_samples_10sec.csv
- /Users/lijie/Desktop/work/attack/dataset/SWaT/extracted/attack_samples_1sec/attack_samples_1sec.csv
- /Users/lijie/Desktop/work/attack/dataset/SWaT/extracted/attack_samples_3sec/attack_samples_3sec.csv
- /Users/lijie/Desktop/work/attack/dataset/SWaT/extracted/benign_samples_10sec/benign_samples_10sec.csv
- /Users/lijie/Desktop/work/attack/dataset/SWaT/extracted/benign_samples_1sec/benign_samples_1sec.csv
- /Users/lijie/Desktop/work/attack/dataset/SWaT/extracted/benign_samples_3sec/benign_samples_3sec.csv
