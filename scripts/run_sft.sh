# scripts/run_sft.sh
"""
这个脚本是一个非常典型的 分布式训练启动命令。
在深度学习（尤其是大语言模型微调）中，使用这种方式而不是直接用 python train.py 是为了处理复杂的计算资源调度。
"""
"""
accelerate launch 可以自动检测硬件环境，并自动检测分布式训练所需要的各种环境变量
--config_file configs/ds_z3_config.json 指向一个配置文件。这里的 ds_z3 通常代表 DeepSpeed ZeRO-3。这是一种极度节省显存的技术，允许你在普通的显卡上训练超大规模的模型。
src/training/train_sft.py 这才是真正包含模型加载、数据处理和训练逻辑的 Python 源代码。
"""
accelerate launch --config_file configs/ds_z3_config.json src/training/train_sft.py