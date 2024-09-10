#!/bin/bash

while getopts m:g: flag
do
    case "${flag}" in
        m) mfrc=${OPTARG};;
        g) goemotions=${OPTARG};;
    esac
done

for model in meta-llama--Llama-2-7b-chat-hf meta-llama--Llama-2-70b-chat-hf meta-llama--Meta-Llama-3-8B-Instruct meta-llama--Meta-Llama-3-70B-Instruct; do

python scripts/compare_distributions.py GoEmotions --root-dir $goemotions --emotion-clustering-json $goemotions/emotion_clustering.json \
    --out logs/analysis/cot/goemotions-$model \
    --experiments \
        ./logs/GoEmotions/$model-cot-annotator-True-None-15-shot_0 \
        ./logs/GoEmotions/$model-cot-True-annotator-True-None-15-shot_0 \
        ./logs/GoEmotions/$model-cot-True-annotator-True-distribution-15-shot_0 \
        ./logs/GoEmotions/$model-cot-False-annotator-True-distribution-15-shot_0 \
        ./logs/GoEmotions/$model-annotator-True-distribution-25-shot_0 \
        ./logs/GoEmotions/$model-cot-annotator-True-None-15-shot_1 \
    --alternative CoT CoT-True-None CoT-True-Distribution CoT-False-Distribution ICL-prior Diff-Chain

done


for model in gpt-3.5-turbo gpt-4o-mini-2024-07-18; do

python scripts/compare_distributions.py GoEmotions --root-dir $goemotions --emotion-clustering-json $goemotions/emotion_clustering.json \
    --out logs/analysis/cot/goemotions-$model \
    --experiments \
        ./logs/GoEmotionsOpenAI/$model-cot-annotator-True-None-15-shot_0 \
        ./logs/GoEmotionsOpenAI/$model-cot-True-annotator-True-None-15-shot_0 \
        ./logs/GoEmotionsOpenAI/$model-cot-True-annotator-True-distribution-15-shot_0 \
        ./logs/GoEmotionsOpenAI/$model-cot-False-annotator-True-distribution-15-shot_0 \
        ./logs/GoEmotionsOpenAI/$model-annotator-True-distribution-45-shot_0 \
        ./logs/GoEmotionsOpenAI/$model-cot-annotator-True-None-15-shot_1 \
    --alternative CoT CoT-True-None CoT-True-Distribution CoT-False-Distribution ICL-prior Diff-Chain

done


for model in meta-llama--Llama-2-7b-chat-hf meta-llama--Llama-2-70b-chat-hf meta-llama--Meta-Llama-3-8B-Instruct meta-llama--Meta-Llama-3-70B-Instruct; do

python scripts/compare_distributions.py MFRC --root-dir $mfrc \
    --out logs/analysis/cot/mfrc-$model \
    --experiments \
        ./logs/MFRC/$model-cot-annotator-True-None-15-shot_0 \
        ./logs/MFRC/$model-cot-True-annotator-True-None-15-shot_0 \
        ./logs/MFRC/$model-cot-True-annotator-True-distribution-15-shot_0 \
        ./logs/MFRC/$model-cot-False-annotator-True-distribution-15-shot_0 \
        ./logs/MFRC/$model-annotator-True-distribution-25-shot_0 \
        ./logs/MFRC/$model-cot-annotator-True-None-15-shot_1 \
    --alternative CoT CoT-True-None CoT-True-Distribution CoT-False-Distribution ICL-prior Diff-Chain

done


for model in gpt-3.5-turbo gpt-4o-mini-2024-07-18; do

python scripts/compare_distributions.py MFRC --root-dir $mfrc \
    --out logs/analysis/cot/mfrc-$model \
    --experiments \
        ./logs/MFRCOpenAI/$model-cot-annotator-True-None-15-shot_0 \
        ./logs/MFRCOpenAI/$model-cot-True-annotator-True-None-15-shot_0 \
        ./logs/MFRCOpenAI/$model-cot-True-annotator-True-distribution-15-shot_0 \
        ./logs/MFRCOpenAI/$model-cot-False-annotator-True-distribution-15-shot_0 \
        ./logs/MFRCOpenAI/$model-annotator-True-distribution-45-shot_0 \
        ./logs/MFRCOpenAI/$model-cot-annotator-True-None-15-shot_1 \
    --alternative CoT CoT-True-None CoT-True-Distribution CoT-False-Distribution ICL-prior Diff-Chain

done