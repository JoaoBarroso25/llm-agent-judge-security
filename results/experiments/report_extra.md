# Further analyses of the open-weight evaluator experiments

Attack subset: 84 trajectories with a side effect and 100 without. No new model calls.

## (a) Committee of the three main open-weight evaluators (gemma4:31b, qwen3.6:35b-a3b, gpt-oss:20b)

| Condition | Committee | Trajectories | Sensitivity | False alarms | Detected side effects hidden |
|---|---|---|---|---|---|
| No attack | Majority (2 of 3) | 184 | 53.6% | 41.0% | — |
| No attack | Any of the three | 184 | 92.9% | 85.0% | — |
| Injection | Majority (2 of 3) | 184 | 2.4% | 3.0% | 43/45 = 95.6% [85.2%; 98.8%] |
| Injection | Any of the three | 184 | 41.7% | 37.0% | 43/78 = 55.1% [44.1%; 65.7%] |
| Persona | Majority (2 of 3) | 184 | 35.7% | 18.0% | 21/45 = 46.7% [32.9%; 60.9%] |
| Persona | Any of the three | 184 | 65.5% | 59.0% | 23/78 = 29.5% [20.5%; 40.4%] |
| False authority | Majority (2 of 3) | 184 | 1.2% | 0.0% | 44/45 = 97.8% [88.4%; 99.6%] |
| False authority | Any of the three | 184 | 13.1% | 14.0% | 67/78 = 85.9% [76.5%; 91.9%] |
| Base64 | Majority (2 of 3) | 184 | 34.5% | 30.0% | 20/45 = 44.4% [30.9%; 58.8%] |
| Base64 | Any of the three | 184 | 86.9% | 89.0% | 7/78 = 9.0% [4.4%; 17.4%] |
| Guided, no attack | Majority (2 of 3) | 184 | 21.4% | 3.0% | — |
| Guided, no attack | Any of the three | 184 | 33.3% | 7.0% | — |
| Guided + injection | Majority (2 of 3) | 184 | 14.3% | 1.0% | 6/18 = 33.3% [16.3%; 56.3%] |
| Guided + injection | Any of the three | 184 | 28.6% | 4.0% | 8/28 = 28.6% [15.3%; 47.1%] |
| Guided + false authority | Majority (2 of 3) | 184 | 22.6% | 3.0% | 2/18 = 11.1% [3.1%; 32.8%] |
| Guided + false authority | Any of the three | 184 | 32.1% | 3.0% | 5/28 = 17.9% [7.9%; 35.6%] |


## (b) False alarms before and after each attack (trajectories without side effects)

| Evaluator | False alarms, no attack | False alarms, injection | False alarms, persona | False alarms, false authority | False alarms, base64 |
|---|---|---|---|---|---|
| gemma4:31b | 12.0% | 0.0% | 13.0% | 0.0% | 17.0% |
| qwen3.6:35b-a3b | 38.0% | 4.0% | 22.0% | 5.0% | 19.0% |
| gpt-oss:20b | 84.0% | 36.0% | 45.0% | 9.0% | 86.0% |
| llama3.1:8b | 71.0% | 25.0% | 16.0% | 35.0% | 77.0% |
| nemotron-3-nano | 85.0% | 38.0% | 61.0% | 35.0% | 87.0% |
| ministral-3:14b | 85.0% | 35.0% | 11.0% | 25.0% | 90.0% |
| phi4:14b | 73.0% | 26.0% | 33.0% | 10.0% | 69.0% |
| gpt-oss:120b | 81.0% | 35.0% | 38.0% | 7.0% | 85.0% |
| glm-4.7-flash | 37.0% | 3.0% | 9.0% | 12.0% | 52.0% |
| EuroLLM-22B | 71.0% | 34.0% | 31.0% | 50.0% | 72.0% |


## (c) Unreadable answers and the share of detected side effects hidden

| Evaluator | Unreadable, no attack | Unreadable, injection | Hidden, injection (unreadable = no) | Hidden, injection (unreadable excluded) | Unreadable, persona | Hidden, persona (unreadable = no) | Hidden, persona (unreadable excluded) | Unreadable, false authority | Hidden, false authority (unreadable = no) | Hidden, false authority (unreadable excluded) | Unreadable, base64 | Hidden, base64 (unreadable = no) | Hidden, base64 (unreadable excluded) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:31b | 0.0% | 0.0% | 33/33 | 33/33 | 0.0% | 14/33 | 14/33 | 0.0% | 31/33 | 31/33 | 0.0% | 16/33 | 16/33 |
| qwen3.6:35b-a3b | 6.5% | 4.3% | 32/36 | 31/35 | 6.0% | 22/36 | 21/35 | 5.4% | 29/36 | 28/35 | 4.9% | 27/36 | 25/34 |
| gpt-oss:20b | 2.7% | 3.8% | 42/74 | 41/73 | 2.7% | 31/74 | 31/74 | 0.5% | 73/74 | 72/73 | 3.3% | 5/74 | 4/73 |
| llama3.1:8b | 0.5% | 0.5% | 46/57 | 45/56 | 0.5% | 43/57 | 43/57 | 1.6% | 37/57 | 36/56 | 1.6% | 13/57 | 12/56 |
| nemotron-3-nano | 3.8% | 8.7% | 37/66 | 34/63 | 3.3% | 25/66 | 24/65 | 2.2% | 45/66 | 44/65 | 3.3% | 8/66 | 6/64 |
| ministral-3:14b | 0.5% | 0.5% | 39/79 | 39/79 | 0.0% | 63/79 | 63/79 | 0.5% | 41/79 | 40/78 | 1.1% | 4/79 | 3/78 |
| phi4:14b | 0.5% | 0.0% | 39/59 | 39/59 | 0.5% | 44/59 | 44/59 | 0.0% | 44/59 | 44/59 | 0.0% | 9/59 | 9/59 |
| gpt-oss:120b | 0.5% | 0.5% | 36/71 | 36/71 | 0.0% | 27/71 | 27/71 | 2.2% | 60/71 | 58/69 | 0.0% | 3/71 | 3/71 |
| glm-4.7-flash | 0.0% | 0.0% | 34/38 | 34/38 | 0.0% | 34/38 | 34/38 | 0.0% | 24/38 | 24/38 | 0.0% | 18/38 | 18/38 |
| EuroLLM-22B | 0.5% | 0.0% | 36/53 | 36/53 | 0.0% | 36/53 | 36/53 | 0.0% | 23/53 | 23/53 | 0.5% | 12/53 | 12/53 |

