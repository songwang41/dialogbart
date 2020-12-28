# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path
# run ./finetune.sh --help to see all the possible options
python finetune.py \
    --learning_rate=3e-5 \
    --fp16 \
    --gpus 4 \
    --do_train \
    --do_predict \
    --n_val 2000 \
    --val_check_interval 0.2 \
    --data_dir $DATA_DIR/train_data  
    --train_batch_size=8     \
    --eval_batch_size=16     \
    --num_train_epochs 6     \
    --max_target_length=60 \
    --val_max_target_length=60 \
    --test_max_target_length=100     \
    --output_dir $DATA_DIR/checkpoints_v2  \
    --model_name_or_path $DATA_DIR/../distilbart-xsum-12-3-dialog 2>&1 | tee $DATA_DIR/training_log_2.txt