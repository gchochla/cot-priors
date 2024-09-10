#!/bin/bash

while getopts d: flag
do
    case "${flag}" in
        d) data=${OPTARG};;
    esac
done

# ICL & priors
python scripts/llm_prompting_clsf.py GoEmotions --root-dir $data \
    --train-split train --test-split dev test --annotation-mode both \
    --text-preprocessor false --emotion-clustering-json $data/emotion_clustering.json \
    --system ' ' --instruction $'Classify the following inputs into none, one, or multiple the following emotions per input: {labels}.\n' \
    --incontext $'Input: {text}\nEmotion(s): {label}\n' \
    --shot 5 25 45 --model-name meta-llama/Meta-Llama-3-8B-Instruct meta-llama/Llama-2-7b-chat-hf \
    --cache-dir /project/shrikann_35/llm-shared --train-keep-same-examples --test-keep-same-examples \
    --train-debug-ann 3 --max-new-tokens 10 --seed 0 1 2 --test-debug-len 100 \
    --keep-one-after-filtering true --label-mode _None_ distribution \
    --load-in-4bit --device cuda:0 --logging-level debug \
    --alternative {model_name_or_path}-annotator-{keep_one_after_filtering}-{label_mode}-{shot}-shot

# CoT
python scripts/prompting/llm_prompting_clsf.py GoEmotions --root-dir $data \
    --train-split train --test-split dev test --annotation-mode both \
    --text-preprocessor false --emotion-clustering-json $data/emotion_clustering.json \
    --system ' ' --instruction $'Classify the following inputs into none, one, or multiple the following emotions per input: {labels}.\n' \
    --incontext $'Input: {text}\n\nReasoning: {cot}\n\nEmotion(s): {label}\n\n' \
    --shot 5 15 --model-name meta-llama/Meta-Llama-3-8B-Instruct meta-llama/Llama-2-7b-chat-hf \
    --cache-dir /project/shrikann_35/llm-shared --train-keep-same-examples --test-keep-same-examples \
    --train-debug-ann 3 --max-new-tokens 80 --seed 0 1 2 --test-debug-len 100 --keep-one-after-filtering true \
    --cot-csv ./files/goemotions_cot.csv --load-in-4bit --accelerate --logging-level debug \
    --alternative {model_name_or_path}-cot-annotator-{keep_one_after_filtering}-{label_mode}-{shot}-shot


# CoT priors
python scripts/prompting/llm_prompting_clsf.py GoEmotions --root-dir $data \
    --train-split train --test-split dev test --annotation-mode both \
    --text-preprocessor false --emotion-clustering-json $data/emotion_clustering.json \
    --system ' ' --instruction $'Classify the following inputs into none, one, or multiple the following emotions per input: {labels}.\n' \
    --incontext $'Input: {text}\n\nReasoning: {cot}\n\nEmotion(s): {label}\n\n' \
    --shot 5 15 --model-name meta-llama/Meta-Llama-3-8B-Instruct meta-llama/Llama-2-7b-chat-hf \
    --cache-dir /project/shrikann_35/llm-shared --train-keep-same-examples --test-keep-same-examples \
    --train-debug-ann 3 --max-new-tokens 80 --seed 0 1 2 --test-debug-len 100 \
    --keep-one-after-filtering true --cot-csv ./files/goemotions_cot.csv \
    { --cot-randomize false --label-mode distribution } { --cot-randomize true --label-mode _None_ distribution } \
    --load-in-4bit --accelerate --logging-level debug \
    --alternative {model_name_or_path}-cot-{cot_randomize}-annotator-{keep_one_after_filtering}-{label_mode}-{shot}-shot
