# scripts/run_sft.sh

# accelerate launch --use_deepspeed --deepspeed_config_file configs/ds_z3_config.json src/training/train_sft.py ...

# 这种方式不会调用 DeepSpeed
accelerate launch --num_processes 1 src/training/train_sft.py