#!/bin/bash

while getopts d: flag
do
    case "${flag}" in
        d) data=${OPTARG};;
    esac
done

python scripts/prompting/api_prompting_clsf.py MFRC --root-dir $data \
    --train-split train --test-split dev test --annotation-mode both \
    --text-preprocessor false \
    --system ' ' --instruction $'Classify the following inputs into none, one, or multiple the following moral foundations per input: {labels}.\n' \
    --incontext $'Input: {text}\n\nReasoning: {cot}\n\nMoral Foundation(s): {label}\n\n' \
    --shot 5 15 --model-name gpt-4o-mini-2024-07-18 gpt-3.5-turbo \
    --train-keep-same-examples --test-keep-same-examples \
    --train-debug-ann 3 --max-new-tokens 80 --seed 0 1 2 --test-debug-len 100 \
    --keep-one-after-filtering true --cot-csv ./files/mfrc_cot.csv \
    --logging-level debug --alternative {model_name}-cot-annotator-{keep_one_after_filtering}-{label_mode}-{shot}-shot

python scripts/prompting/api_prompting_clsf.py MFRC --root-dir $data \
    --train-split train --test-split dev test --annotation-mode both \
    --text-preprocessor false \
    --system ' ' --instruction $'Classify the following inputs into none, one, or multiple the following moral foundations per input: {labels}.\n' \
    --incontext $'Input: {text}\n\nReasoning: {cot}\n\nMoral Foundation(s): {label}\n\n' \
    --shot 5 15 --model-name gpt-4o-mini-2024-07-18 gpt-3.5-turbo \
    --train-keep-same-examples --test-keep-same-examples \
    --train-debug-ann 3 --max-new-tokens 80 --seed 0 1 2 --test-debug-len 100 \
    --keep-one-after-filtering true --cot-csv ./files/mfrc_cot.csv \
    \{ --cot-randomize false --label-mode distribution \} \{ --cot-randomize true --label-mode _None_ distribution \} \
    --logging-level debug --alternative {model_name}-cot-{cot_randomize}-annotator-{keep_one_after_filtering}-{label_mode}-{shot}-shot
