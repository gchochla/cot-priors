# Larger Language Models Don't Care How You Think: Why Chain-of-Thought Prompting Fails in Subjective Tasks

This repo contains the official implementation of [Larger Language Models Don't Care How You Think: Why Chain-of-Thought Prompting Fails in Subjective Tasks](), currently under review.

## Abstract

> In-Context Learning (ICL) in Large Language Models (LLM) has emerged as the dominant technique for performing natural language tasks, as it does not require updating the model parameters with gradient-based methods. ICL promises to "adapt" the LLM to perform the present task at a competitive or state-of-the-art level at a fraction of the computational cost. ICL can be augmented by incorporating the reasoning process to arrive at the final label explicitly in the prompt, a technique called Chain-of-Thought (CoT) prompting. However, recent work has found that ICL relies mostly on the retrieval of task priors and less so on "learning" to perform tasks, especially for complex subjective domains like emotion and morality, where priors ossify posterior predictions. In this work, we examine whether "enabling" reasoning also creates the same behavior in LLMs, wherein the format of CoT retrieves *reasoning priors* that remain relatively unchanged despite the evidence in the prompt. We find that, surprisingly, CoT indeed suffers from the same posterior collapse as ICL for larger language models.

## Installation

This repo uses `Python 3.10` (type hints, for example, won't work with some previous versions). After you create and activate your virtual environment (with conda, venv, etc), install local dependencies with:

```bash
pip install -e .[dev]
```

## Data preparation

For CoT, we have included our annotations in `./files`. If you want to generate your own results, you can use `./scripts/cot-csv-creation.sh` to generate a CSV file with the text, add a column `cot`, and include your reasonings there.

To run the GoEmotions experiments, we recommend using the emotion pooling we set up based on the hierarchical clustering (besides, the bash scripts are set up for it). To do so, create the file `emotion_clustering.json` under the root folder of the dataset with the following contents:

```JSON
{
    "joy": [
        "amusement",
        "excitement",
        "joy",
        "love"
    ],
    "optimism": [
        "desire",
        "optimism",
        "caring"
    ],
    "admiration": [
        "pride",
        "admiration",
        "gratitude",
        "relief",
        "approval",
        "realization"
    ],
    "surprise": [
        "surprise",
        "confusion",
        "curiosity"
    ],
    "fear": [
        "fear",
        "nervousness"
    ],
    "sadness": [
        "remorse",
        "embarrassment",
        "disappointment",
        "sadness",
        "grief"
    ],
    "anger": [
        "anger",
        "disgust",
        "annoyance",
        "disapproval"
    ]
}
```

For MFRC, please create a folder for the dataset (even though we use HuggingFace `datasets` for it), and copy the file `./files/splits.yaml` to that directory.

## Run experiments

Experiments are logged with [legm](https://github.com/gchochla/legm), so refer to the documentation there for an interpretation of the resulting `logs` folder, but navigating should be intuitive enough with some trial and error. Note that some bash scripts have arguments, which are self-explanatory. Make sure to run scripts from the root directory of this repo.

Also, you should create a `.env` file with your OpenAI key if you want to perform experiments with the GPTs.

```bash
OPENAI_API_KEY=<your-openai-key>
```

Then, proceed to run the experiments in `./scripts`:

- `goemotions.sh --data /path/to/goemotions` will run HuggingFace models on GoEmotions, including CoT and ICL
- `mfrc-openai-cot.sh --data /path/to/mfrc` will run OpenAI models on MFRC with CoT
- For all other use cases, you can recreate the experiments by combining elements of the above two files.

One these experiments have been run, you can generate the analysis files by running `./scripts/cot_priors_comparison.sh`, and then the figures with `scripts/figures.py` (e.g., `python scripts/figures.py cot_vs_icl logs/analysis/cot logs`, or `python scripts/cot-prior/figures.py cot_priors logs/analysis/cot logs/analysis/cot`).
