# CICIDS2017 Modern SOTA Comparison @ FPR=0.15

| method | type | target_fpr | auc | ap | f1 | recall | precision | test_fpr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TCN-GAN | Ours | 0.1500 | 0.9464 | 0.9351 | 0.8783 | 0.9678 | 0.8040 | 0.1729 |
| GANomaly | SOTA | 0.1500 | 0.7409 | 0.7706 | 0.6585 | 0.7237 | 0.6041 | 0.3474 |
| TranAD | SOTA | 0.1500 | 0.5660 | 0.5786 | 0.4849 | 0.4181 | 0.5771 | 0.2244 |
| DeepSVDD | SOTA | 0.1500 | 0.4782 | 0.4664 | 0.3462 | 0.2419 | 0.6091 | 0.1137 |

说明：GANomaly、DeepSVDD、TranAD 均已补齐 target FPR=0.15，避免高召回表缺少现代异常检测 baseline。
