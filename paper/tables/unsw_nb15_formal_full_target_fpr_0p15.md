# UNSW-NB15 Formal Full Comparison @ FPR=0.15

| dataset | method | type | target_fpr | auc | ap | f1 | recall | precision | test_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNSW-NB15 | IsolationForest | Unsupervised | 0.1500 | 0.8327 | 0.8430 | 0.8272 | 0.9856 | 0.7127 | 0.4926 |
| UNSW-NB15 | TranAD | SOTA | 0.1500 | 0.8467 | 0.8666 | 0.8178 | 0.9219 | 0.7349 | 0.4124 |
| UNSW-NB15 | DeepSVDD | SOTA | 0.1500 | 0.7334 | 0.7047 | 0.8177 | 0.9965 | 0.6932 | 0.5466 |
| UNSW-NB15 | TCN-GAN | Ours | 0.1500 | 0.8017 | 0.7549 | 0.7928 | 0.9037 | 0.7062 | 0.4660 |
| UNSW-NB15 | GANomaly | SOTA | 0.1500 | 0.9703 | 0.9721 | 0.7613 | 0.9996 | 0.6147 | 0.7768 |
| UNSW-NB15 | LSTM-AE | Deep | 0.1500 | 0.7964 | 0.8249 | 0.7205 | 0.7243 | 0.7168 | 0.3548 |
| UNSW-NB15 | OneClassSVM | Unsupervised | 0.1500 | 0.7576 | 0.8197 | 0.6178 | 0.5081 | 0.7879 | 0.1696 |
| UNSW-NB15 | LSTM-AD | Deep | 0.1500 | 0.6723 | 0.6976 | 0.5546 | 0.4480 | 0.7280 | 0.2075 |

说明：本表使用 UNSW-NB15 官方 `Training and Testing Sets`，仅使用数值型特征，窗口大小 128、stride 16、窗口异常比例阈值 0.15。TCN-GAN 行来自 `results/cross_dataset_formal_unsup/unsw_nb15_tcn/20260420_230046/ours.json`。
