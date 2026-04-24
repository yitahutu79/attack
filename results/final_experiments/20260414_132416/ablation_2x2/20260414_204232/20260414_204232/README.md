# TCN-GAN autotune run

- Results CSV: `attack/results/final_experiments/20260414_132416/ablation_2x2/20260414_204232/20260414_204232/tcn_gan_autotune_results.csv`
- Best (by `calib_f1`): window=128 stride=16 score=fused
  - score_alpha=0.24
  - out_json: `attack/results/final_experiments/20260414_132416/ablation_2x2/20260414_204232/20260414_204232/eval_w128_s16_fused.json`
  - xai_report: `attack/results/xai_tcn/eval_w128_s16_fused_w128_s16_fused_a0.24/xai_report.json`
  - xai_time_plot: `attack/results/xai_tcn/eval_w128_s16_fused_w128_s16_fused_a0.24/xai_time_importance.png`
  - xai_feat_plot: `attack/results/xai_tcn/eval_w128_s16_fused_w128_s16_fused_a0.24/xai_feature_importance.png`

## All runs (sorted)

| rank | window | stride | score | alpha | AUC | AP | calib_f1 | calib_rec | logs |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 128 | 16 | fused | 0.24 | 0.9469153830004308 | 0.9289104171454587 | 0.8258218695420811 | 0.7260532270229502 | `attack/results/final_experiments/20260414_132416/ablation_2x2/20260414_204232/20260414_204232/eval_w128_s16_fused.log` |
