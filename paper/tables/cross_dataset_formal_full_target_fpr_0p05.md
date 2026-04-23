# Cross-Dataset Formal Full Comparison @ FPR=0.05

| dataset | method | type | target_fpr | auc | ap | f1 | recall | precision | test_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SWaT | TCN-GAN | Ours | 0.0500 | 0.9904 | 0.9977 | 0.9950 | 0.9900 | 1.0000 | 0.0000 |
| SWaT | OneClassSVM | Unsupervised/Deep | 0.0500 | 0.9926 | 0.9982 | 0.9913 | 0.9827 | 1.0000 | 0.0000 |
| SWaT | TranAD | SOTA | 0.0500 | 0.9800 | 0.9953 | 0.9885 | 0.9800 | 0.9972 | 0.0090 |
| SWaT | IsolationForest | Unsupervised/Deep | 0.0500 | 0.9861 | 0.9964 | 0.9768 | 0.9599 | 0.9943 | 0.0179 |
| SWaT | LSTM-AE | Unsupervised/Deep | 0.0500 | 0.9800 | 0.9953 | 0.9867 | 0.9800 | 0.9935 | 0.0209 |
| SWaT | LSTM-AD | Unsupervised/Deep | 0.0500 | 0.8680 | 0.9661 | 0.8988 | 0.8288 | 0.9817 | 0.0507 |
| SWaT | DeepSVDD | SOTA | 0.0500 | 0.8485 | 0.9314 | 0.8636 | 0.9918 | 0.7647 | 1.0000 |
| TON_IoT | TCN-GAN | Ours | 0.0500 | 0.5556 | 0.3804 | 0.1794 | 0.1101 | 0.4838 | 0.0497 |
| TON_IoT | LSTM-AD | Unsupervised/Deep | 0.0500 | 0.4848 | 0.2864 | 0.1130 | 0.0748 | 0.2314 | 0.1052 |
| TON_IoT | IsolationForest | Unsupervised/Deep | 0.0500 | 0.3986 | 0.2555 | 0.1409 | 0.0993 | 0.2423 | 0.1316 |
| TON_IoT | OneClassSVM | Unsupervised/Deep | 0.0500 | 0.3754 | 0.2694 | 0.2442 | 0.2110 | 0.2897 | 0.2192 |
| TON_IoT | DeepSVDD | SOTA | 0.0500 | 0.5835 | 0.3705 | 0.3791 | 0.3823 | 0.3759 | 0.2688 |
| TON_IoT | TranAD | SOTA | 0.0500 | 0.4406 | 0.2827 | 0.3058 | 0.3210 | 0.2921 | 0.3296 |
| TON_IoT | LSTM-AE | Unsupervised/Deep | 0.0500 | 0.6125 | 0.3586 | 0.4541 | 0.5611 | 0.3814 | 0.3854 |
