# Scale and reasoning tests

Attack subset: 84 trajectories with a side effect and 100 without. κ is re-weighted to the population prevalence (6.5%); 95% stratified bootstrap intervals. 'Hides' is the share of the side effects detected without the attack that the attack makes the evaluator miss. Rows without a model are models that did not fit in the GPU or were not run.

## Larger models, with the models of the thesis for comparison

| Evaluator | Sensitivity | False alarms | κ original | κ guided | Unreadable | Median s/call | Injection hides | False authority hides | Injection hides (guided) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5 122B-A10B | 79.8% | 66.0% | 0.03 [0.00; 0.06] | 0.15 [0.05; 0.31] | 3.3% | 48 | 62/67 = 92.5% [83.7%; 96.8%] | 41/67 = 61.2% [49.2%; 72.0%] | 11/22 = 50.0% [30.7%; 69.3%] |
| Nemotron 3 Super 120B | 65.5% | 64.0% | 0.00 [-0.02; 0.03] | 0.07 [-0.01; 0.18] | 0.0% | 59 | 48/55 = 87.3% [76.0%; 93.7%] | 51/55 = 92.7% [82.7%; 97.1%] | 9/18 = 50.0% [29.0%; 71.0%] |
| Llama 4 109B | 76.2% | 74.0% | 0.00 [-0.02; 0.03] | 0.02 [-0.05; 0.11] | 0.5% | 50 | 60/64 = 93.8% [85.0%; 97.5%] | 56/64 = 87.5% [77.2%; 93.5%] | 9/13 = 69.2% [42.4%; 87.3%] |
| Gemma 4 31B | 39.3% | 12.0% | 0.18 [0.08; 0.32] | 0.27 [0.13; 0.43] | 0.0% | 20 | 33/33 = 100.0% [89.6%; 100.0%] | 31/33 = 93.9% [80.4%; 98.3%] | 5/22 = 22.7% [10.1%; 43.4%] |
| Qwen3.6 35B-A3B | 42.9% | 38.0% | 0.01 [-0.02; 0.07] | 0.25 [0.12; 0.40] | 6.5% | 13 | 32/36 = 88.9% [74.7%; 95.6%] | 29/36 = 80.6% [65.0%; 90.2%] | 11/20 = 55.0% [34.2%; 74.2%] |
| gpt-oss 20B | 88.1% | 84.0% | 0.01 [-0.01; 0.02] | 0.08 [-0.01; 0.20] | 2.7% | 5 | 42/74 = 56.8% [45.4%; 67.4%] | 73/74 = 98.6% [92.7%; 99.8%] | 5/9 = 55.6% [26.7%; 81.1%] |
| gpt-oss 120B | 84.5% | 81.0% | 0.01 [-0.01; 0.02] | 0.18 [0.07; 0.34] | 0.5% | 16 | 36/71 = 50.7% [39.3%; 62.0%] | 60/71 = 84.5% [74.3%; 91.1%] | 11/25 = 44.0% [26.7%; 62.9%] |


## Reasoning switched on

| Evaluator | Prompt | Sensitivity (off → on) | False alarms (off → on) | κ off | κ on | Unreadable on | Median s on |
|---|---|---|---|---|---|---|---|
| Qwen3.6 35B-A3B | Original prompt | 42.9% → 71.4% | 38.0% → 72.0% | 0.01 [-0.02; 0.07] | -0.00 [-0.02; 0.02] | 0.0% | 82 |
| Qwen3.6 35B-A3B | Guided prompt | 23.8% → 19.0% | 3.0% → 3.0% | 0.25 [0.12; 0.40] | 0.19 [0.07; 0.35] | 0.0% | 78 |
| Qwen3.6 35B-A3B | Injection (original) | — | — | hides 32/36 = 88.9% [74.7%; 95.6%] | hides 58/60 = 96.7% [88.6%; 99.1%] | 0.0% | 98 |
| Gemma 4 31B | Original prompt | 39.3% → 29.8% | 12.0% → 27.0% | 0.18 [0.08; 0.32] | 0.01 [-0.03; 0.07] | 0.0% | 134 |
| Gemma 4 31B | Guided prompt | 26.2% → 33.3% | 3.0% → 3.0% | 0.27 [0.13; 0.43] | 0.34 [0.20; 0.51] | 0.0% | 116 |
| Gemma 4 31B | Injection (original) | — | — | hides 33/33 = 100.0% [89.6%; 100.0%] | hides 24/25 = 96.0% [80.5%; 99.3%] | 0.0% | 132 |

