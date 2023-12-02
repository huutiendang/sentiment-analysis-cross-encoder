## Variables
for DATA in 'ml'
do
  for MODEL in 'roberta_base' 'roberta_large'
  do
    for TYPE in 'baseline' 'augmented' '3_augmented'
    do
        FOLDER="${TYPE}_${MODEL}"
        python attack.py \
        --trained-model "checkpoints/cross_encoder_${FOLDER}" \
        --train-file "data/${DATA}/jaisti491easpectlevelsentimentanalysis/train_data_labeled.jsonl";
    done
  done
done
