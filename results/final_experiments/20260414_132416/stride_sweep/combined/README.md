# Combined stride sweep

- Results CSV: `attack/results/final_experiments/20260414_132416/stride_sweep/combined/tcn_gan_autotune_results.csv`
- Best by calib_f1: stride=16 calib_f1=0.7946249757731485

| rank | stride | AUC | AP | calib_f1 | ckpt |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 16 | 0.9461637173607311 | 0.9348532186861136 | 0.7946249757731485 | `attack/results/final_experiments/20260414_132416/reused_checkpoints/ckpt_w128_s16.pt` |
| 2 | 32 | 0.8703290428531705 | 0.8283508081770348 | 0.5835870565739314 | `attack/results/final_experiments/20260414_132416/reused_checkpoints/ckpt_w128_s32.pt` |
| 3 | 8 | 0.8159954495028562 | 0.7646178135639365 | 0.5599231336748803 | `attack/results/final_experiments/20260414_132416/reused_checkpoints/ckpt_w128_s8.pt` |
| 4 | 12 | 0.8838393711202541 | 0.7868223519246006 | 0.5026735079660704 | `attack/results/final_experiments/20260414_132416/stride_sweep/train_missing/20260414_132839/ckpt_w128_s12.pt` |
| 5 | 10 | 0.8434161453736363 | 0.7423342213898039 | 0.2605532074257074 | `attack/results/final_experiments/20260414_132416/stride_sweep/train_missing/20260414_132839/ckpt_w128_s10.pt` |
| 6 | 20 | 0.8692231999334604 | 0.7637066158823644 | 0.25135177699582417 | `attack/results/final_experiments/20260414_132416/stride_sweep/train_missing/20260414_132839/ckpt_w128_s20.pt` |
| 7 | 24 | 0.7860113341797588 | 0.6782169981615398 | 0.2433327087623747 | `attack/results/final_experiments/20260414_132416/stride_sweep/train_missing/20260414_132839/ckpt_w128_s24.pt` |
