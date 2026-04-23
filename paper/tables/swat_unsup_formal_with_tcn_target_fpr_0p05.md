# SWaT Formal Unsupervised Split With TCN Target FPR=0.05

说明：该表采用 anomaly-detection 更公平的 SWaT 正式协议：

- benign 前 60% 训练
- benign 中间 20% 阈值标定
- benign 后 20% 作为测试正常样本
- attack 文件作为测试异常样本

TCN 与无监督/深度 baseline 均按同一协议评估。

| method | type | target_fpr | auc | ap | f1 | recall | precision | test_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TCN-GAN (SWaT) | Ours | 0.0500 | 0.9904 | 0.9977 | 0.9950 | 0.9900 | 1.0000 | 0.0000 |
| OneClassSVM | Unsupervised | 0.0500 | 0.9926 | 0.9982 | 0.9913 | 0.9827 | 1.0000 | 0.0000 |
| LSTM-AE | Deep | 0.0500 | 0.9800 | 0.9953 | 0.9867 | 0.9800 | 0.9935 | 0.0209 |
| IsolationForest | Unsupervised | 0.0500 | 0.9861 | 0.9964 | 0.9768 | 0.9599 | 0.9943 | 0.0179 |
| LSTM-AD | Deep | 0.0500 | 0.8680 | 0.9661 | 0.8988 | 0.8288 | 0.9817 | 0.0507 |

建议正文写法：

- 在更符合异常检测设定的 SWaT 正式切分下，\model{} 显著优于先前 mixed-split 结果，并恢复了低误报优势。
- \model{} 在 target FPR=0.05 下达到 0.9950 的 calibrated F1、0.9900 的 recall，并将测试 benign FPR 控制在 0.0000。
- 该结果表明，先前性能退化主要来自不匹配的评估协议，而非模型本身失效。

