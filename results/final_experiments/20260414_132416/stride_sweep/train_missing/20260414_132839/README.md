# TCN-GAN autotune run

- Results CSV: `attack/results/final_experiments/20260414_132416/stride_sweep/train_missing/20260414_132839/tcn_gan_autotune_results.csv`
- Best (by `calib_f1`): window=128 stride=12 score=fused
  - score_alpha=0.20
  - out_json: `attack/results/final_experiments/20260414_132416/stride_sweep/train_missing/20260414_132839/eval_w128_s12_fused.json`

## All runs (sorted)

| rank | window | stride | score | alpha | AUC | AP | calib_f1 | calib_rec | logs |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 128 | 12 | fused | 0.2 | 0.8838393711202541 | 0.7868223519246006 | 0.5026735079660704 | 0.3759948289096271 | `attack/results/final_experiments/20260414_132416/stride_sweep/train_missing/20260414_132839/eval_w128_s12_fused.log` |
| 2 | 128 | 10 | fused | 0.2 | 0.8434161453736363 | 0.7423342213898039 | 0.2605532074257074 | 0.16396202275941013 | `attack/results/final_experiments/20260414_132416/stride_sweep/train_missing/20260414_132839/eval_w128_s10_fused.log` |
| 3 | 128 | 20 | fused | 0.2 | 0.8692231999334604 | 0.7637066158823644 | 0.25135177699582417 | 0.16119041206571502 | `attack/results/final_experiments/20260414_132416/stride_sweep/train_missing/20260414_132839/eval_w128_s20_fused.log` |
| 4 | 128 | 24 | fused | 0.2 | 0.7860113341797588 | 0.6782169981615398 | 0.2433327087623747 | 0.15738870485578085 | `attack/results/final_experiments/20260414_132416/stride_sweep/train_missing/20260414_132839/eval_w128_s24_fused.log` |
