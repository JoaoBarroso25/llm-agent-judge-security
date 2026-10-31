# Security Evaluation of LLM-Based Agents

Code, results and full text of the MSc dissertation **_Security Evaluation of LLM-Based Agents: Robustness Against Jailbreaks, Adversarial Attacks and Multi-Agent Manipulation_**, by João Alexandre Antunes da Costa Barroso, supervised by Manuel Fernando da Silva Rodrigues (Master's in Informatics Engineering, University of Minho, 2026).

📄 **Dissertation:** [`thesis/Barroso_2026_Security_Evaluation_of_LLM-Based_Agents.pdf`](thesis/Barroso_2026_Security_Evaluation_of_LLM-Based_Agents.pdf)

## What the work asks

When an LLM-based agent acts on the web, another LLM is often used to judge what it did. This work asks whether such evaluators notice when the agent does something it was not asked to do (an unsafe *side effect*), and whether the agent being judged can manipulate the model that judges it.

It combines two parts:

1. **A secondary analysis of [AgentRewardBench](https://huggingface.co/datasets/McGill-NLP/agent-reward-bench)** (Lù et al., 2025): 1,302 expert-annotated trajectories of four web agents on five benchmarks, and the verdicts of 11 published LLM evaluators (14,322 evaluator × trajectory pairs).
2. **Eleven controlled experiments (E1–E11)** with 3 closed and 13 open-weight evaluator models. They cover prompts, inputs, prompt injections, jailbreaks, defences, larger models, reasoning, committees, an adaptive attacker who knows the defence, and a structural defence.

## Main findings

- **Unsafe behaviour is present without any adversary.** 6.5% of the trajectories contain side effects [95% CI 5.2–7.9%], and 51.9% contain repetition loops.
- **The published evaluators detect side effects at about chance level.** Their Cohen's κ ranges from −0.01 to 0.13, against 0.65 between two experts. They recognise task success and loops reasonably well.
- **Their false alarms follow the length of the run, not the risk.** False alarms rise from 11.2% to 66.0% as runs get longer (OR 8.0, against 1.47 for real side effects). Their safety verdict is entangled with their success verdict, and their errors are shared, so committees do not help.
- **The evaluated agent can manipulate its evaluator.**
  - An instruction written by the agent in its own reasoning hides 37% of the side effects detected by Gemini 3.7 Flash, and 49–100% on ten open-weight models.
  - A "system override" jailbreak hides up to 99%.
  - The injection still works when a third party plants it on the web page (49–88%), at temperature 0, against committees, and against models of up to 125B parameters.
- **A guideline-based prompt helps but can be defeated.** An attacker who knows it still hides 65% of the detections of one of the two strongest open-weight evaluators.
- **A structural defence removes the attacker's channel.** In this defence, a quarantined reader turns the record into a constrained table and the deciding evaluator never reads the agent's text. Attacks fall to the level of run-to-run noise, but detection does not improve.

## Repository contents

| Path | Contents |
|---|---|
| `thesis/` | The dissertation (PDF). |
| `code/` | All analysis and experiment code (Python 3.12). |
| `results/` | Reports (`*.md`), result tables (`tables/*.csv`) and, in `experiments/`, every evaluator response collected in the experiments. |

**Secondary analysis**
- `analysis.py`: RQ1–RQ4.
- `analysis_extended.py`: mechanisms and robustness.
- `analysis_third.py`: surface baselines, context and triage.
- `examples.py`: annotated examples.

**Experiments**
- `experiments.py`: E1–E9; definition of every condition and attack text.
- `noise_guided.py`, `adaptive_attack.py`: E10.
- `structured_defence.py`: E11.
- `run_*.py` and `run_*.sh`: supervisors for long runs.

**Analysis of the experiments**
- `experiments_analysis.py`, `experiments_extra.py`, `experiments_scale_analysis.py`, `adaptive_analysis.py`, `structured_analysis.py`.

**Output**
- `figures.py`, `figures_extra.py`: figures.
- `typst_tables.py`: tables of the dissertation (Typst).

### The experiment records

`results/experiments/raw.jsonl.gz` holds every evaluator response (51,672 valid responses). Each line contains:
- the condition, benchmark, agent, task and model;
- the parsed verdicts;
- the full text of the answer;
- the token counts.

The cache is keyed by condition and trajectory, so a request is never repeated. All the analyses of the experiments read this file and send no request to any model.

The other files in that folder are:
- `structured_reader.jsonl`: the tables written by the quarantined reader of E11.
- `adaptive_*.json`: the attack texts and search states of E10 and E11.
- `sample.csv`: the sample of trajectories used in the experiments.

## Reproducing the results

The secondary analysis and all the analyses of the experiments run on a laptop in a few minutes. Every random step uses seed 2026.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. AgentRewardBench annotations and published verdicts (about 4 GB, not redistributed here)
bash code/download_data.sh

# 2. Secondary analysis (RQ1–RQ4)
python3 code/analysis.py
python3 code/analysis_extended.py
python3 code/analysis_third.py
python3 code/examples.py

# 3. Analyses of the experiments, from the stored responses (no model is called)
gunzip -k results/experiments/raw.jsonl.gz
python3 code/experiments_analysis.py
python3 code/experiments_extra.py
python3 code/experiments_scale_analysis.py
python3 code/adaptive_analysis.py
python3 code/structured_analysis.py

# 4. Figures and tables of the dissertation (written to Tese/figures and Tese/generated)
python3 code/figures.py
python3 code/figures_extra.py
python3 code/typst_tables.py
```

The reports are written to `results/*.md` and `results/experiments/*_report.md`, and the tables to `results/tables/`.

### Running new evaluator calls (optional)

Running the experiments again is only needed to extend them. The scripts skip every call already stored in `raw.jsonl`.

- **Gemini models.** Install `google-genai` and set `GOOGLE_CLOUD_PROJECT`, with Google Cloud credentials for Vertex AI.
- **Open-weight models.** Serve them with [Ollama](https://ollama.com) and set `OLLAMA_URL` (default `http://localhost:11434`). If the server needs a bearer token, set `OLLAMA_KEY_FILE` to a file that contains it.

To stop a long run without losing anything, create the file `results/experiments/STOP`.

## Data and ethics

- **Data.** The AgentRewardBench data belong to their authors and are downloaded from their public release; they are not redistributed here.
- **Safety of the experiments.** No agent was executed and no real system was touched. The experiments only send recorded trajectories to an evaluator model and read its verdict.
- **Attack texts.** The adversarial texts are deliberately simple and target no deployed system. They are published so that future evaluators can be tested against them.
- **Which models received what.** The jailbreaks were sent only to public open-weight models.

## Citation

```bibtex
@mastersthesis{barroso2026security,
  author = {Barroso, Jo{\~a}o Alexandre Antunes da Costa},
  title  = {Security Evaluation of {LLM}-Based Agents: Robustness Against Jailbreaks, Adversarial Attacks and Multi-Agent Manipulation},
  school = {University of Minho},
  year   = {2026},
  type   = {Master's dissertation},
  url    = {https://github.com/JoaoBarroso25/llm-agent-judge-security}
}
```

## Licence

- **Code:** MIT licence (see [`LICENSE`](LICENSE)).
- **Dissertation and results:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), the licence of the dissertation.
