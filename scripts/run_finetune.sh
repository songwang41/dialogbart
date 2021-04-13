ln -s /mount/csi_modeling/data data
python finetune.py --learning_rate 1e-5 --fp16 --gpus 4 \
  --do_train \
  --val_check_interval 0.5 --n_val 80000 \
  --data_dir data/train_dev_conv_sum/chat_summary_segment_v7_v2/train_data_small_sales_gpt3labels_v2/ \
  --train_batch_size 8 --eval_batch_size 8 --num_train_epochs 10 \
  --max_target_length 82 --val_max_target_length 82 --test_max_target_length 120  \
  --num_workers 2 \
  --eval_max_gen_length 82 \
  --output_dir data/train_dev_conv_sum/chat_summary_segment_v7_v2/train2m_dialogbart_new_short_min2max62_b128_train_shuf_gpt3sales_v2 \
  --model_name_or_path data/train_dev_conv_sum/chat_summary_segment_v7_v2/train2m_dialogbart_new_short_min2max62_b128_train_shuf/best_tfmr \
  --accumulate_grad_batches 4 --overwrite_output_dir \
  --save_top_k 10
