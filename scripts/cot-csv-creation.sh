#!/bin/bash

while getopts m:g: flag
do
    case "${flag}" in
        m) mfrc=${OPTARG};;
        g) goemotions=${OPTARG};;
    esac
done


python scripts/get_random_ids_from_train.py GoEmotions \
    --root-dir $goemotions --annotation-mode both \
    --text-preprocessor false --shot 5 15 --seed {0..2} --output-filename $goemotions/annotator_demos.txt \
    --train-debug-ann 3 --train-keep-same-examples --keep-one-after-filtering false \
    --emotion-clustering-json $goemotions/emotion_clustering.json


python scripts/get_random_ids_from_train.py GoEmotions \
    --root-dir $goemotions --annotation-mode both \
    --text-preprocessor false --shot 5 15 --seed {0..2} --output-filename $goemotions/aggregate_demos.txt \
    --train-debug-ann 3 --train-keep-same-examples --keep-one-after-filtering true \
    --emotion-clustering-json $goemotions/emotion_clustering.json


python scripts/get_random_ids_from_train.py MFRC \
    --root-dir $mfrc --annotation-mode both \
    --text-preprocessor false --shot 5 15 --seed {0..2} --output-filename $mfrc/annotator_demos.txt \
    --train-debug-ann 3 --train-keep-same-examples --keep-one-after-filtering false


python scripts/get_random_ids_from_train.py MFRC \
    --root-dir $mfrc --annotation-mode both \
    --text-preprocessor false --shot 5 15 --seed {0..2} --output-filename $mfrc/aggregate_demos.txt \
    --train-debug-ann 3 --train-keep-same-examples --keep-one-after-filtering true

python scripts/mix_csvs.py --i $mfrc/annotator_demos.csv $mfrc/aggregate_demos.csv --o $mfrc/demos.csv
python scripts/mix_csvs.py --i $goemotions/annotator_demos.csv $goemotions/aggregate_demos.csv --o $goemotions/demos.csv