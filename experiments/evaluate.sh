# Variables
for TYPE in 'baseline' 'augmented' '3_augmented'
do
    for NAME in 'roberta_base' 'roberta_large'
    do
        TRAIN_MODEL="checkpoints/cross_encoder_${TYPE}_${NAME}"
        # Train model with noise full data
        python evaluate.py \
            --type "$TYPE" \
            --model-name "$NAME" \
            --trained-model "$TRAIN_MODEL" \
            --test-file data/ml/jaisti491easpectlevelsentimentanalysis/test_data.jsonl;
    done
done
