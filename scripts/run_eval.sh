export CUDA_VISIBLE_DEVICES=0
cd ~/work/train_dev_conv_sum/dialogbart_folder/dialogbart/scripts
ln -s ~/mount/csi_modeling/data data
MODEL_DIR=data/train_dev_conv_sum/gpt3_data_bart/train200k_checkpoints_dialogbart-min11-max100_b96-20epochs/
DATA_DIR=data/skylight_summary/train7/train_data_v4/
MIN_LEN=2
MAX_LEN=82
B=32
output_pred_file=pred_skylight694_b${B}_min${MIN_LEN}_max${MAX_LEN}.txt
output_score_file=pred_skylight694_b${B}_min${MIN_LEN}_max${MAX_LEN}_score.json

python run_eval.py $MODEL_DIR/best_tfmr \
$DATA_DIR/test.source  \
$MODEL_DIR/$output_pred_file \
--task summarization \
--score_path $MODEL_DIR/$output_score_file \
--reference_path $DATA_DIR/test.target \
--device cuda \
--fp16 \
--bs $B \
--max_length $MAX_LEN \
--min_length $MIN_LEN

rm data