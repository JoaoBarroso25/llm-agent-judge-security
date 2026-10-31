# Evaluator experiments (gemini-3.7-flash, Vertex AI)
Sample: 350 trajectories (84 with side effects). Calls: 40455 valid (endpoint: {'uminho-ollama': 36772, 'global': 3638, 'eu': 45}); total cost $15.09; thinking level 'low'.
Calls and cost by model: gemini-3.5-flash-lite: 368 calls, $0.59; gemini-3.7-flash: 2947 calls, $12.98; gemini-3.8-flash: 368 calls, $1.52; gemma4:31b: 4296 calls, $0.00; glm-4.7-flash:latest: 3412 calls, $0.00; gpt-oss:120b: 3412 calls, $0.00; gpt-oss:20b: 4296 calls, $0.00; hf.co/mradermacher/EuroLLM-22B-Instruct-2512-i1-GGUF:Q4_K_M: 3412 calls, $0.00; llama3.1:8b: 3412 calls, $0.00; ministral-3:14b: 3412 calls, $0.00; nemotron-3-nano:latest: 3412 calls, $0.00; phi4:14b: 3412 calls, $0.00; qwen3.6:35b-a3b: 4296 calls, $0.00.

## Detection of side effects by condition

Sensitivity and false alarms as observed in the sample; precision and κ re-weighted to the population prevalence (6.5%). Unreadable verdicts count as 'no'.

| condition | trajectories | with side effect | sensitivity | false alarms | precision (population) | κ (population) [95% CI] | unreadable |
|---|---|---|---|---|---|---|---|
| GPT-4o mini, actions only (published) | 350 | 84 | 75.0% | 66.9% | 7.2% | 0.01 [-0.00; 0.04] | 0 |
| Original prompt | 350 | 84 | 22.6% | 3.8% | 29.3% | 0.21 [0.10; 0.33] | 0 |
| Original prompt (second run) | 100 | 20 | 30.0% | 1.2% | 62.4% | 0.38 [0.13; 0.60] | 1 |
| Guided prompt | 349 | 84 | 28.6% | 3.4% | 36.7% | 0.28 [0.17; 0.41] | 0 |
| Guided prompt, no agent narrative | 350 | 84 | 21.4% | 3.4% | 30.4% | 0.21 [0.11; 0.32] | 0 |
| Original + false self-report | 184 | 84 | 23.8% | 2.0% | 45.1% | 0.28 [0.14; 0.43] | 0 |
| Original + injection to the evaluator | 184 | 84 | 14.3% | 0.0% | 100.0% | 0.24 [0.13; 0.35] | 0 |
| Guided + false self-report | 184 | 84 | 29.8% | 2.0% | 50.7% | 0.34 [0.20; 0.51] | 0 |
| Guided + injection to the evaluator | 183 | 83 | 28.9% | 2.0% | 49.9% | 0.33 [0.20; 0.49] | 0 |
| GPT-4o mini, with accessibility tree (published) | 350 | 84 | 69.0% | 63.9% | 6.9% | 0.01 [-0.01; 0.03] | 0 |
| Original prompt, with accessibility tree | 337 | 75 | 21.3% | 3.8% | 27.8% | 0.20 [0.09; 0.31] | 0 |
| Original + defence by instruction | 182 | 84 | 25.0% | 3.1% | 36.0% | 0.26 [0.13; 0.41] | 0 |
| Original + defence by instruction + injection | 182 | 83 | 21.7% | 1.0% | 59.7% | 0.29 [0.16; 0.43] | 0 |
| Original + datamarking | 183 | 84 | 20.2% | 3.0% | 31.6% | 0.21 [0.09; 0.36] | 0 |
| Original + datamarking + injection | 179 | 83 | 22.9% | 2.1% | 43.1% | 0.27 [0.14; 0.42] | 0 |
| gemini-3.8-flash, original prompt | 184 | 84 | 26.2% | 4.0% | 31.1% | 0.24 [0.12; 0.40] | 0 |
| gemini-3.8-flash, original + injection | 184 | 84 | 19.0% | 2.0% | 39.7% | 0.22 [0.10; 0.37] | 0 |
| gemini-3.5-flash-lite, original prompt | 184 | 84 | 25.0% | 20.0% | 7.9% | 0.03 [-0.03; 0.10] | 1 |
| gemini-3.5-flash-lite, original + injection | 184 | 84 | 0.0% | 0.0% | — | 0.00 [0.00; 0.00] | 1 |

## Paired comparisons (same trajectories)

| comparison | trajectories | n | flagged (first) | flagged (second) | only first flags | only second flags | McNemar p |
|---|---|---|---|---|---|---|---|
| guided vs original | with side effect | 84 | 22.6% | 28.6% | 1 | 6 | 0.125 |
| guided vs original | without side effect | 265 | 3.8% | 3.4% | 2 | 1 | 1 |
| guided without narrative vs original | with side effect | 84 | 22.6% | 21.4% | 5 | 4 | 1 |
| guided without narrative vs original | without side effect | 266 | 3.8% | 3.4% | 2 | 1 | 1 |
| without vs with narrative (guided) | with side effect | 84 | 28.6% | 21.4% | 7 | 1 | 0.0703 |
| without vs with narrative (guided) | without side effect | 265 | 3.4% | 3.4% | 1 | 1 | 1 |
| Gemini vs published GPT-4o mini | with side effect | 84 | 75.0% | 22.6% | 47 | 3 | 3.71e-11 |
| Gemini vs published GPT-4o mini | without side effect | 266 | 66.9% | 3.8% | 168 | 0 | 5.35e-51 |
| attack: false self-report (original) | with side effect | 84 | 22.6% | 23.8% | 1 | 2 | 1 |
| attack: false self-report (original) | without side effect | 100 | 4.0% | 2.0% | 2 | 0 | 0.5 |
| attack: injection (original) | with side effect | 84 | 22.6% | 14.3% | 7 | 0 | 0.0156 |
| attack: injection (original) | without side effect | 100 | 4.0% | 0.0% | 4 | 0 | 0.125 |
| attack: false self-report (guided) | with side effect | 84 | 28.6% | 29.8% | 1 | 2 | 1 |
| attack: false self-report (guided) | without side effect | 99 | 2.0% | 2.0% | 0 | 0 | 1 |
| attack: injection (guided) | with side effect | 83 | 27.7% | 28.9% | 0 | 1 | 1 |
| attack: injection (guided) | without side effect | 99 | 2.0% | 2.0% | 0 | 0 | 1 |
| with vs without accessibility tree | with side effect | 75 | 21.3% | 21.3% | 4 | 4 | 1 |
| with vs without accessibility tree | without side effect | 262 | 3.8% | 3.8% | 2 | 2 | 1 |
| Gemini vs GPT-4o mini, accessibility tree | with side effect | 75 | 66.7% | 21.3% | 37 | 3 | 1.95e-08 |
| Gemini vs GPT-4o mini, accessibility tree | without side effect | 262 | 63.4% | 3.8% | 156 | 0 | 2.19e-47 |
| defence by instruction vs none (no attack) | with side effect | 84 | 22.6% | 25.0% | 1 | 3 | 0.625 |
| defence by instruction vs none (no attack) | without side effect | 98 | 3.1% | 3.1% | 1 | 1 | 1 |
| datamarking vs none (no attack) | with side effect | 84 | 22.6% | 20.2% | 3 | 1 | 0.625 |
| datamarking vs none (no attack) | without side effect | 99 | 4.0% | 3.0% | 2 | 1 | 1 |
| attack: injection (defence by instruction) | with side effect | 83 | 24.1% | 21.7% | 3 | 1 | 0.625 |
| attack: injection (defence by instruction) | without side effect | 97 | 3.1% | 1.0% | 2 | 0 | 0.5 |
| attack: injection (datamarking) | with side effect | 83 | 20.5% | 22.9% | 2 | 4 | 0.688 |
| attack: injection (datamarking) | without side effect | 95 | 2.1% | 2.1% | 0 | 0 | 1 |
| attack: injection (gemini-3.8-flash) | with side effect | 84 | 26.2% | 19.0% | 6 | 0 | 0.0312 |
| attack: injection (gemini-3.8-flash) | without side effect | 100 | 4.0% | 2.0% | 2 | 0 | 0.5 |
| attack: injection (gemini-3.5-flash-lite) | with side effect | 84 | 25.0% | 0.0% | 21 | 0 | 9.54e-07 |
| attack: injection (gemini-3.5-flash-lite) | without side effect | 100 | 20.0% | 0.0% | 20 | 0 | 1.91e-06 |

## Attacks: side effects hidden from the evaluator

Share of the side effects detected without the attack that the evaluator no longer flags when the agent's last reasoning contains the attack text.

| judge | attack | side effects detected without attack | hidden by the attack | sensitivity before → after | false alarms before → after | judged successful before → after | McNemar p (side effects) |
|---|---|---|---|---|---|---|---|
| gemini-3.7-flash, original | False self-report | 19/84 | 1/19 = 5.3% [0.9%; 24.6%] | 22.6% → 23.8% | 4.0% → 2.0% | 27.2% → 26.1% | 1 |
| gemini-3.7-flash, original | Injection to the evaluator | 19/84 | 7/19 = 36.8% [19.1%; 59.0%] | 22.6% → 14.3% | 4.0% → 0.0% | 27.2% → 25.5% | 0.0156 |
| gemini-3.7-flash, guided | False self-report | 24/84 | 1/24 = 4.2% [0.7%; 20.2%] | 28.6% → 29.8% | 2.0% → 2.0% | — | 1 |
| gemini-3.7-flash, guided | Injection to the evaluator | 23/83 | 0/23 = 0.0% [0.0%; 14.3%] | 27.7% → 28.9% | 2.0% → 2.0% | — | 1 |
| gemini-3.7-flash, original + defence by instruction | Injection to the evaluator | 20/83 | 3/20 = 15.0% [5.2%; 36.0%] | 24.1% → 21.7% | 3.1% → 1.0% | 25.6% → 25.0% | 0.625 |
| gemini-3.7-flash, original + datamarking | Injection to the evaluator | 17/83 | 2/17 = 11.8% [3.3%; 34.3%] | 20.5% → 22.9% | 2.1% → 2.1% | 26.4% → 25.8% | 0.688 |
| gemini-3.8-flash, original | Injection to the evaluator | 22/84 | 6/22 = 27.3% [13.2%; 48.2%] | 26.2% → 19.0% | 4.0% → 2.0% | 21.7% → 20.7% | 0.0312 |
| gemini-3.5-flash-lite, original | Injection to the evaluator | 21/84 | 21/21 = 100.0% [84.5%; 100.0%] | 25.0% → 0.0% | 20.0% → 0.0% | 37.2% → 39.9% | 9.54e-07 |

## Run-to-run variability (original prompt, 100 trajectories run twice)

Same side-effect verdict in both runs: 97.0%; κ between runs: 0.78; flips yes→no 2, no→yes 1.

## Evaluators compared (detection and attacks)

Hidden = detected side effects (without attack) that the evaluator no longer flags with the attack.

| model | weights | trajectories | unreadable | sensitivity | false alarms | κ (population) | κ guided, no narrative | κ with accessibility tree | agreement between runs | κ guided | hidden: injection | hidden: injection (guided) | hidden: injection, defence by instruction | hidden: injection, datamarking | hidden: false self-report | κ temperature 0 | κ attack subset | sensitivity attack subset | false alarms attack subset | sensitivity temperature 0 | false alarms temperature 0 | agreement default vs temperature 0 | hidden: persona | hidden: false authority | hidden: base64 | hidden: persona (guided) | hidden: authority (guided) | hidden: base64 (guided) | hidden: injection, temperature 0 | hidden: false authority, temperature 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemini-3.7-flash | closed | 350 | 0 | 22.6% | 3.8% | 0.21 | 0.21 | 0.20 | 97.0% | 0.28 | 7/19 | 0/23 | 3/20 | 2/17 | 1/19 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| gemini-3.8-flash | closed | 184 | 0 | 26.2% | 4.0% | 0.24 | — | — | — | — | 6/22 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| gemini-3.5-flash-lite | closed | 184 | 1 | 25.0% | 20.0% | 0.03 | — | — | — | — | 21/21 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| gemma4:31b | open | 350 | 0 | 39.3% | 17.3% | 0.12 | 0.15 | 0.13 | 83.0% | 0.23 | 33/33 | 5/22 | 25/25 | 30/33 | 12/33 | 0.04 | 0.18 | 39.3% | 12.0% | 27.4% | 19.0% | 79.9% | 14/33 | 31/33 | 16/33 | 0/22 | 1/22 | 1/22 | 22/23 | 21/23 |
| glm-4.7-flash:latest | open | 184 | 0 | 45.2% | 37.0% | 0.03 | 0.12 | 0.03 | 56.0% | 0.11 | 34/38 | 15/18 | 34/37 | 37/44 | 17/38 | — | — | — | — | — | — | — | 34/38 | 24/38 | 18/38 | 17/18 | 14/18 | 8/18 | — | — |
| gpt-oss:120b | open | 184 | 1 | 84.5% | 81.0% | 0.01 | 0.09 | 0.00 | 94.0% | 0.18 | 36/71 | 11/25 | 44/68 | 35/67 | 6/71 | — | — | — | — | — | — | — | 27/71 | 60/71 | 3/71 | 10/25 | 10/25 | 9/25 | — | — |
| gpt-oss:20b | open | 350 | 9 | 88.1% | 80.5% | 0.01 | 0.09 | 0.00 | 92.0% | 0.06 | 42/74 | 5/9 | 40/71 | 34/72 | 9/74 | 0.00 | 0.01 | 88.1% | 84.0% | 89.3% | 86.0% | 91.8% | 31/74 | 73/74 | 5/74 | 4/9 | 3/9 | 2/9 | 38/75 | 56/75 |
| hf.co/mradermacher/EuroLLM-22B-Instruct-2512-i1-GGUF:Q4_K_M | open | 184 | 1 | 63.1% | 71.0% | -0.01 | 0.03 | -0.04 | 84.0% | 0.03 | 36/53 | 27/35 | 29/51 | 31/49 | 32/53 | — | — | — | — | — | — | — | 36/53 | 23/53 | 12/53 | 20/35 | 20/35 | 12/35 | — | — |
| llama3.1:8b | open | 184 | 1 | 67.9% | 71.0% | -0.01 | 0.01 | -0.01 | 84.0% | -0.00 | 46/57 | 25/40 | 44/63 | 41/65 | 47/57 | — | — | — | — | — | — | — | 43/57 | 37/57 | 13/57 | 31/40 | 23/40 | 16/40 | — | — |
| ministral-3:14b | open | 184 | 1 | 94.0% | 85.0% | 0.01 | 0.06 | 0.01 | 97.0% | 0.14 | 39/79 | 9/22 | 36/80 | 41/76 | 7/79 | — | — | — | — | — | — | — | 63/79 | 41/79 | 4/79 | 13/22 | 7/22 | 6/22 | — | — |
| nemotron-3-nano:latest | open | 184 | 7 | 78.6% | 85.0% | -0.01 | 0.03 | 0.01 | 84.0% | 0.06 | 37/66 | 40/60 | 46/74 | 19/73 | 17/66 | — | — | — | — | — | — | — | 25/66 | 45/66 | 8/66 | 28/60 | 34/60 | 27/60 | — | — |
| phi4:14b | open | 184 | 1 | 70.2% | 73.0% | -0.00 | 0.03 | 0.01 | 83.0% | 0.06 | 39/59 | 15/25 | 41/59 | 36/62 | 17/59 | — | — | — | — | — | — | — | 44/59 | 44/59 | 9/59 | 17/25 | 16/25 | 9/25 | — | — |
| qwen3.6:35b-a3b | open | 350 | 21 | 42.9% | 36.1% | 0.02 | 0.16 | 0.00 | 68.0% | 0.23 | 32/36 | 11/20 | 33/36 | 30/38 | 15/36 | 0.02 | 0.01 | 42.9% | 38.0% | 45.2% | 39.0% | 69.0% | 22/36 | 29/36 | 27/36 | 4/20 | 5/20 | 7/20 | 33/38 | 29/38 |
