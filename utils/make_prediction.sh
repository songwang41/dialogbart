export CUDA_VISIBLE_DEVICES=3

DATA_DIR=/home/sonwang/mount/csi_modeling/data/train_dev_kb_multilingual/train_data_v2/released_modelV6_trainv2_20kXum/kb_multilingual_12_2_model_16fp/
MODEL_DIR=/home/sonwang/mount/csi_modeling/data/train_dev_kb_multilingual/train_data_v4_withEng_new/mbart-large-cc25-xsum_epoch10_b32_len80_roRO_continued_ckt25_kb_len80_epoch20/trainV4Eng_ckpt29_distil_12_12_ckpt16_final_model_v7/
postfix='released_modelV6_trainv2_20kXum'
MIN_LEN=12
MAX_LEN=82
B=32
output_pred_file=pred_tranV4test_b${B}_min${MIN_LEN}_max${MAX_LEN}_$postfix.txt
output_score_file=pred_tranV4test_b${B}_min${MIN_LEN}_max${MAX_LEN}_score_$postfix.json



python run_eval.py $MODEL_DIR \
$DATA_DIR/test.source  \
$MODEL_DIR/../$output_pred_file \
--task summarization \
--score_path $MODEL_DIR/../$output_score_file \
--reference_path $DATA_DIR/test.target \
--device cuda \
--fp16 \
--bs $B \
--max_length $MAX_LEN \
--min_length $MIN_LEN

files2rouge $MODEL_DIR/../$output_pred_file $DATA_DIR/test.target > $MODEL_DIR/../file2rouge_${output_pred_file} 2>&1