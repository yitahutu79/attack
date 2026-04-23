# Figure 1: Overall Framework of Attentive TCN-WGAN-GP

Create a clean academic SVG figure for an IEEE-style paper.

The figure title must be:
Overall Framework of Attentive TCN-WGAN-GP

Use a white background, flat vector style, 3 to 4 soft colors, clear arrows, and no decorative icons. Do not include citation bubbles. Do not let arrows cross text. Keep the diagram readable when scaled to a two-column paper width. Use English labels only.

Main pipeline from left to right:

1. Network Flow Records
   - CICIDS2017
   - ordered flow features

2. Preprocessing
   - cleaning + normalization
   - temporal order preserved

3. Sliding Window Construction
   - X in R^{128 x 77}
   - W = 128, stride = 16
   - visually show overlapping windows

4. Attentive TCN-WGAN-GP Detector
   This is the central and largest module.
   Include the following internal components:
   - TCN Generator G
   - Synthetic Window X_hat
   - TCN Critic D
   - Attention Pooling
   - WGAN-GP Objective

   Scientific logic inside the detector:
   - latent noise z goes into TCN Generator G
   - Generator G produces Synthetic Window X_hat
   - both the real window X and Synthetic Window X_hat are scored by TCN Critic D
   - Attention Pooling learns temporal evidence weights
   - WGAN-GP Objective connects to Generator G and Critic D
   - WGAN-GP is a training objective, not an attention module

5. Fused Anomaly Score
   - S = alpha S_D + (1-alpha) S_F
   - alpha = 0.24
   - critic evidence + feature deviation

6. Threshold Calibration
   - target FPR = 0.05 / 0.15

7. Online Alarm
   - normal / anomalous window

Lower support row:

8. Evaluation Protocol
   - baseline comparison
   - ablation study
   - multi-seed stability
   - runtime analysis

9. XAI Analysis
   - feature attribution
   - temporal attribution
   - attention weights
   XAI is post-hoc explanation after training, not a training input.

10. Paper Claims
   - accurate detection
   - real-time feasibility
   - interpretable alarms

Connections:
- Network Flow Records -> Preprocessing -> Sliding Window Construction -> Attentive TCN-WGAN-GP Detector -> Fused Anomaly Score -> Threshold Calibration -> Online Alarm
- Sliding Window Construction or Detector -> Evaluation Protocol
- Detector or Fused Anomaly Score -> XAI Analysis
- Evaluation Protocol -> Paper Claims
- XAI Analysis -> Paper Claims

Strict requirements:
- no spelling mistakes
- no duplicate Evaluation Protocol
- no duplicate Paper Claims
- no duplicate Attention Pooling
- no arrows through formulas
- no unrelated icons such as locks, cloud, server, shield, or people
- keep formulas short and readable
