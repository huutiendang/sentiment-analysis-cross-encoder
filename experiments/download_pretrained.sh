output_folder='../checkpoints/'
link_ids=(
    '1pFMk0yH40987mfDl3OVWbkeTEWd85lJg' # 3 aug base
    '1T50TgZToFbevOUmfdVTRRy808-xsgOBH' # 3 aug large
    '1nwXUP6Y3aBYeyQl4zJ1CfJcZ-egiSXzv' # aug base
    '1SBVE_ROE_v_uWRw2Yk-Ir4lCpFmI2NlX' # aug large
    '1WsiOo3oddQoHGzdQbTD3dUM91iIS_ttg' # baseline base
    '1DwEJgVy7R8dFHxCoRd9b-Rfb5gVvCcs1' # baseline large
)
name=(
  'cross_encoder_3_augmented_roberta_base'
  'cross_encoder_3_augmented_roberta_large'
  'cross_encoder_augmented_roberta_base'
  'cross_encoder_augmented_roberta_large'
  'cross_encoder_baseline_roberta_base'
  'cross_encoder_baseline_roberta_large'
)


for ((i=0; i<${#link_ids[@]}; i++)); do
    FILE="${names[$i]}.zip"
    gdown "https://drive.google.com/uc?id=${link_ids[$i]}" -O "$FILE"
    unzip -q "$FILE" -d "$output_folder"
    rm "$FILE"
    rm -rf "$output_folder/__MACOSX"
done