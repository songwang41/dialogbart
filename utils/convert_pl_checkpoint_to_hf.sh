model_dir='root_model_dir'
python convert_pl_checkpoint_to_hf.py --pl_ckpt_path $model_dir/ckpts/val_avg_rouge2=60.0774-step_count=23.ckpt \
--hf_src_model_dir $model_dir/best_tfmr --save_path $model_dir/ckt23