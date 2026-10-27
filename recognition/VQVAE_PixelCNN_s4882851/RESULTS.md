# Campaign results

Complete: **True**
Validation-only chosen VQ run: `vq_l1_s0`

| Run | Epochs | Seconds/epoch | Peak GB | Validation metric |
|---|---:|---:|---:|---:|
| gan_s0 | 50 | 8.65 | 1.01 | 0.00000 |
| gan_s1 | 50 | 8.65 | 1.01 | 0.00000 |
| prior_grid8_s0 | 40 | 3.11 | 0.17 | 3.60220 |
| prior_init_s0 | 40 | 7.87 | 0.68 | 3.33092 |
| prior_k128_s0 | 40 | 6.79 | 0.68 | 2.92428 |
| prior_l1_s0 | 40 | 7.85 | 0.68 | 3.45483 |
| prior_main_s0 | 40 | 7.87 | 0.68 | 3.43864 |
| prior_main_s1 | 40 | 7.86 | 0.68 | 3.59565 |
| prior_wide7 | 40 | 4.54 | 0.28 | 3.50964 |
| vq_grid8_s0 | 100 | 5.96 | 0.77 | 0.78503 |
| vq_init_s0 | 100 | 8.02 | 1.03 | 0.92867 |
| vq_k128_s0 | 100 | 7.82 | 1.03 | 0.91835 |
| vq_l1_s0 | 100 | 8.21 | 1.03 | 0.93164 |
| vq_main_s0 | 100 | 8.07 | 1.00 | 0.93111 |
| vq_main_s1 | 100 | 8.01 | 1.00 | 0.93329 |

## Held-out results
| Run | Reconstruction SSIM | VQ+prior KID | GAN KID |
|---|---:|---:|---:|
| eval_grid8_s0 | 0.81504 | 0.35742080211639404 | 0.48895201086997986 |
| eval_init_s0 | 0.93906 | 0.4722023010253906 | 0.48895201086997986 |
| eval_k128_s0 | 0.92912 | 0.3478020429611206 | 0.48895201086997986 |
| eval_l1_s0 | 0.94247 | 0.42179176211357117 | 0.48895201086997986 |
| eval_main_s0 | 0.94140 | 0.37335464358329773 | 0.48895201086997986 |
| eval_main_s1 | 0.94318 | 0.43083932995796204 | 0.2829633355140686 |
| eval_wide7 | 0.94140 | 0.44766607880592346 | 0.48895201086997986 |
| temperature_t0.7 | 0.93111 | 0.3070141673088074 | None |
| temperature_t1 | 0.93111 | 0.4027966260910034 | None |
| temperature_t1.3 | 0.93111 | 0.6511056423187256 | None |

Missing evaluations: []
Failed/unverified tasks: []

Human annotations and ethical/reliability interpretation remain mandatory.
