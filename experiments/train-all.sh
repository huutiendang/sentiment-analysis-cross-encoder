# Variables
MODEL=roberta_base
DATA=ml
python baseline.py \
    --model-name $MODEL \
    --train-file data/$DATA/jaisti491easpectlevelsentimentanalysis/train_data_labeled.jsonl \
    --train-unlabeled-file data/$DATA/jaisti491easpectlevelsentimentanalysis/train_data_unlabel.jsonl \
    --test-file data/$DATA/jaisti491easpectlevelsentimentanalysis/test_data.jsonl \
    --model-save-path checkpoints/cross_encoder_baseline_$MODEL;

python augmentation.py \
    --model-name $MODEL \
    --train-file data/$DATA/jaisti491easpectlevelsentimentanalysis/train_data_labeled.jsonl \
    --train-labeled data/$DATA/processed/train_labeled_$MODEL.csv;

python re-train.py \
    --model-name $MODEL \
    --train-file data/$DATA/jaisti491easpectlevelsentimentanalysis/train_data_labeled.jsonl \
    --train-augmented data/$DATA/processed/augmentation_$MODEL.csv \
    --train-labeled data/$DATA/processed/train_labeled_$MODEL.csv \
    --test-file data/$DATA/jaisti491easpectlevelsentimentanalysis/test_data.jsonl \
    --model-save-path checkpoints/cross_encoder_3_augmented_$MODEL;