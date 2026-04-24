# Split and Window Statistics

Window label rule: A window is labeled anomalous when the attack-record ratio within that window is >= Anomaly Ratio Threshold.

| Dataset | Split | #Records | #Windows | #Benign Windows | #Anomalous Windows | Attack Types | Window Size | Stride | Anomaly Ratio Threshold |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| CICIDS2017 | raw_total | 2300825 | - | - | - | - | 128 | 16 | 0.15 |
| CICIDS2017 | train | 1595677 | 81522 | 81522 | 0 | - | 128 | 16 | 0.15 |
| CICIDS2017 | calibration(train_benign_reuse) | 1327906 | 81522 | 81522 | 0 | - | 128 | 16 | 0.15 |
| CICIDS2017 | test | 702718 | 43897 | 25335 | 18562 | PortScan:10297; DDoS:8117; Bot:148 | 128 | 16 | 0.15 |
| SWaT | raw_total | 45055 | - | - | - | - | 128 | 16 | 0.15 |
| SWaT | train | 16416 | 1019 | 1019 | 0 | - | 128 | 16 | 0.15 |
| SWaT | calibration | 5472 | 335 | 335 | 0 | - | 128 | 16 | 0.15 |
| SWaT | test | 23167 | 1433 | 335 | 1098 | recon:429; dos:221; ddos:192; malware:102; mitm:92; web:34; bruteforce:28 | 128 | 16 | 0.15 |
| UNSW-NB15 | raw_total | 257673 | - | - | - | - | 128 | 16 | 0.15 |
| UNSW-NB15 | train | 175341 | 2988 | 2988 | 0 | - | 128 | 16 | 0.15 |
| UNSW-NB15 | calibration(train_benign_reuse) | 56000 | 2988 | 2988 | 0 | - | 128 | 16 | 0.15 |
| UNSW-NB15 | test | 82332 | 5138 | 2294 | 2844 | Generic:1596; Exploits:935; Fuzzers:240; DoS:73 | 128 | 16 | 0.15 |

Notes:
- CICIDS2017 calibration windows are reused from benign train windows (no separate calibration file split).
- UNSW-NB15 calibration windows are reused from benign train windows (official split has no standalone calibration file).
- Attack type counts are computed on anomalous windows by dominant attack type per window.
