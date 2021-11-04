# this script works for a single GPU
# you must run the model using 
# checkout kb_sum_multilingual
# --task summarization
# not --decoder_start_token_id 250020


export CUDA_VISIBLE_DEVICES=1
ROOT_DIR=data/train_dev_multilingual_kb
MODEL_DIR=$ROOT_DIR/kb_models_multi/trainv2_xsum_kb_len80/best_tfmr
DATA_DIR=$ROOT_DIR/train_data
DISTILATED_MODEL_DIR=$MODEL_DIR/../distilmbart_model
python distillation.py \
  --teacher $MODEL_DIR --data_dir  $DATA_DIR \
  --tokenizer_name $MODEL_DIR \
  --student_decoder_layers 2 --student_encoder_layers 12 \
  --freeze_encoder --freeze_embeds \
  --learning_rate=3e-4 \
  --do_train \
  --do_predict \
  --fp16 --fp16_opt_level=O1 --gpus 1 \
  --val_check_interval 0.5 --n_val 8000 --eval_beams 2 --length_penalty=0.7 \
  --max_target_length=80 --val_max_target_length=100 --test_max_target_length=100 \
  --eval_max_gen_length 80 \
  --model_name_or_path IGNORED \
  --alpha_hid=3. \
  --train_batch_size=4 --eval_batch_size=4 --gradient_accumulation_steps=8 \
  --sortish_sampler \
  --num_train_epochs=10 \
  --warmup_steps 500 \
  --save_top_k 10 \
  --output_dir $DISTILATED_MODEL_DIR \
  --overwrite_output_dir