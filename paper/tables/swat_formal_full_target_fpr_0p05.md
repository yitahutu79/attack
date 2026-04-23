# SWaT Formal Full Comparison @ FPR=0.05

| dataset | method | type | target_fpr | auc | ap | f1 | recall | precision | test_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SWaT | TCN-GAN | Ours | 0.0500 | 0.9904 | 0.9977 | 0.9950 | 0.9900 | 1.0000 | 0.0000 |
| SWaT | OneClassSVM | Unsupervised/Deep | 0.0500 | 0.9926 | 0.9982 | 0.9913 | 0.9827 | 1.0000 | 0.0000 |
| SWaT | TranAD | SOTA | 0.0500 | 0.9800 | 0.9953 | 0.9885 | 0.9800 | 0.9972 | 0.0090 |
| SWaT | IsolationForest | Unsupervised/Deep | 0.0500 | 0.9861 | 0.9964 | 0.9768 | 0.9599 | 0.9943 | 0.0179 |
| SWaT | LSTM-AE | Unsupervised/Deep | 0.0500 | 0.9800 | 0.9953 | 0.9867 | 0.9800 | 0.9935 | 0.0209 |
| SWaT | LSTM-AD | Unsupervised/Deep | 0.0500 | 0.8680 | 0.9661 | 0.8988 | 0.8288 | 0.9817 | 0.0507 |
| SWaT | DeepSVDD | SOTA | 0.0500 | 0.8485 | 0.9314 | 0.8636 | 0.9918 | 0.7647 | 1.0000 |
