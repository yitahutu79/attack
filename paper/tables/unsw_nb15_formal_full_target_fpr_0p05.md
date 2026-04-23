# UNSW-NB15 Formal Full Comparison @ FPR=0.05

| dataset | method | type | target_fpr | auc | ap | f1 | recall | precision | test_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNSW-NB15 | GANomaly | SOTA | 0.0500 | 0.9703 | 0.9721 | 0.8311 | 0.9972 | 0.7124 | 0.4991 |
| UNSW-NB15 | DeepSVDD | SOTA | 0.0500 | 0.7334 | 0.7047 | 0.8177 | 0.9965 | 0.6932 | 0.5466 |
| UNSW-NB15 | TranAD | SOTA | 0.0500 | 0.8467 | 0.8666 | 0.7061 | 0.6164 | 0.8265 | 0.1604 |
| UNSW-NB15 | TCN-GAN | Ours | 0.0500 | 0.8017 | 0.7549 | 0.6742 | 0.5795 | 0.8059 | 0.1731 |
| UNSW-NB15 | IsolationForest | Unsupervised | 0.0500 | 0.8327 | 0.8430 | 0.6621 | 0.5665 | 0.7967 | 0.1792 |
| UNSW-NB15 | LSTM-AE | Deep | 0.0500 | 0.7964 | 0.8249 | 0.6602 | 0.5383 | 0.8534 | 0.1146 |
| UNSW-NB15 | OneClassSVM | Unsupervised | 0.0500 | 0.7576 | 0.8197 | 0.5298 | 0.3787 | 0.8813 | 0.0632 |
| UNSW-NB15 | LSTM-AD | Deep | 0.0500 | 0.6723 | 0.6976 | 0.2406 | 0.1424 | 0.7744 | 0.0514 |

说明：本表使用 UNSW-NB15 官方 `Training and Testing Sets`，仅使用数值型特征，窗口大小 128、stride 16、窗口异常比例阈值 0.15。
