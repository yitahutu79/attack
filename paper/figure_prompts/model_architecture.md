# Figure 2: Attentive TCN-WGAN-GP Model Architecture

Create a clean academic SVG architecture diagram for an IEEE-style paper.

The figure title must be:
Attentive TCN-WGAN-GP Model Architecture

Use a white background, flat vector style, soft blue, purple, green, yellow, red, and gray. Avoid gradients, shadows, and decorative icons. Do not include citation bubbles. Do not let arrows overlap text. Use English labels only.

The figure should not be a pure text flowchart. It must include visual structures:
- input window matrix
- TCN dilated convolution blocks
- temporal hidden-state sequence
- attention weight bars
- generator path
- synthetic window matrix
- critic score
- fused anomaly score

Input:
- X in R^{128 x 77}
- 128 time steps and 77 flow features

Top path: Real-window critic path

1. Input Window X
   - draw a matrix or heatmap-like grid
   - label: X in R^{128 x 77}

2. TCN Encoder
   - draw several temporal convolution blocks
   - show dilations: d=1, d=2, d=4, d=8
   - label: dilated temporal convolutions

3. Temporal Hidden States
   - draw a sequence of feature vectors
   - label: H = [h_1, h_2, ..., h_W]

4. Attention Pooling
   - draw bars with different heights above hidden states
   - label: learned temporal weights a_t
   - formula: h = sum_t a_t h_t

5. Critic D
   - label: critic score S_D(X)

Bottom path: Generator path

6. Latent Noise
   - label: z ~ p(z)
   - draw random dots or a vector

7. TCN Generator G
   - draw neural network block plus temporal convolution blocks
   - label: linear projection + temporal convolutions

8. Synthetic Window
   - draw a matrix similar to the input window
   - label: X_hat = G(z)

Training objective:

9. WGAN-GP Training Objective
   - connect to Generator G and Critic D
   - show that real window X and synthetic window X_hat are both scored by Critic D
   - label:
     L_D = E[D(X_hat)] - E[D(X)] + lambda GP
     L_G = -E[D(G(z))]
   - WGAN-GP is a training objective, not a network layer

Inference:

10. Feature-Deviation Score
    - label: S_F(X)

11. Fused Inference Score
    - receives S_D(X) and S_F(X)
    - formula:
      S = alpha S_D + (1-alpha) S_F
      alpha = 0.24

12. Decision
    - threshold tau
    - normal / anomalous

Strict requirements:
- no spelling mistakes
- do not draw XAI as a training input
- do not draw WGAN-GP as attention
- keep labels short
- use clear arrows
- keep the layout readable in a paper figure
