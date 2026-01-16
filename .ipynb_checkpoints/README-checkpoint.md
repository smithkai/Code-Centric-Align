两个db文件和output.txt没完全搞明白

目录结构待更新
Code-Centric-Align/
├── data/                        # 数据存放目录（不进入Git版本控制）
│   ├── raw/                     # 原始下载的数据 (如 magicoder_75k.jsonl)
│   ├── processed/               # 清洗、去重、演进后的最终训练数据
│   └── eval/                    # 评测集数据 (HumanEval, MBPP)
├── configs/                     # 存放所有配置文件
│   ├── sft_config.yaml          # LLaMA-Factory 或自定义 SFT 训练参数
│   ├── dpo_config.yaml          # DPO 训练参数
│   └── deepspeed_zero3.json     # DeepSpeed 分布式插件配置
├── src/                         # 核心源代码
│   ├── data_engine/             # 第一阶段：数据工程脚本
│   │   ├── collect.py           # 数据采集与流式下载
│   │   ├── clean.py             # 长度过滤、特殊字符清洗
│   │   ├── dedup.py             # MinHash/LSH 去重实现
│   │   └── evol_instruct.py     # 调用 API 进行指令演进 (Evol)
│   ├── training/                # 第二、四阶段：训练脚本
│   │   ├── train_sft.py         # SFT 启动逻辑
│   │   ├── train_dpo.py         # DPO 启动逻辑
│   │   └── utils.py             # 训练工具类（如计算KL散度）
│   ├── alignment/               # 第三阶段：拒绝采样流水线
│   │   ├── sampler.py           # 模型推理采样 (N个答案)
│   │   └── sandbox.py           # 自动化 Python 代码执行沙箱 (Docker版)
│   └── eval/                    # 第五阶段：模型评测
│       └── evaluate_pass_at_k.py # 计算 Pass@1, Pass@10 指标
├── scripts/                     # 自动化 Shell 脚本 (用于 Linux 集群一键运行)
│   ├── run_sft.sh
│   └── run_dpo.sh
├── notebooks/                   # 用于在本地做 EDA (数据探索分析) 的 Jupyter
├── requirements.txt             # 环境依赖列表
├── .gitignore                   # 忽略大型数据文件和模型权重
└── README.md                    # 项目说明文档
