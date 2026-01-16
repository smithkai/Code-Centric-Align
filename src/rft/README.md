├── post_training         # 新建：存放拒绝采样、DPO 相关核心逻辑
│   ├── rs_engine.py      # 核心：高并发采样引擎
│   ├── rs_validator.py   # 核心：多进程执行验证器
│   └── dpo_trainer.py    # 核心：DPO 训练启动脚本