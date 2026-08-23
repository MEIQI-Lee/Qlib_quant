# Qlib_quant：Qlib 框架在科创板股票中的应用

> 基于微软 Qlib AI 量化投资平台，对科创板（STAR Market）全部股票进行收益率预测的量化研究项目。

## 项目简介

本项目采用微软开源的 **Qlib**（AI-oriented Quantitative Investment Platform）量化投资框架，以上海证券交易所 **科创板（STAR Market）** 全部上市公司股票为研究对象，构建了从数据清洗、特征工程、模型训练到结果评估的完整量化投研流水线。

项目对比了两类主流模型在科创板截面收益预测上的表现：

- **ALSTM（Attention LSTM）**：引入注意力机制的长短期记忆网络，属于 Qlib 内置的深度学习模型
- **LightGBM**：梯度提升决策树，属于经典的树集成学习模型

同时在 `Models.py` 中扩展实现了十余种 PyTorch 自定义时序模型（GRU 系列、TCN、CNN-LSTM/GRU、FEDformer、Informer 等），方便进行更广泛的模型对比实验。

---

## 项目文件结构

```
Qlib_quant/
├── Models.py                    # PyTorch 自定义模型库（15+ 种时序模型）
├── data_clean.ipynb             # 原始数据清洗与预处理 Notebook
├── dump_bin.py                  # CSV 数据转换为 Qlib 二进制格式的脚本
├── qlib框架code.ipynb           # Qlib 框架主流程代码（数据加载/训练/回测）
├── 科创板_lgb.ipynb             # 科创板 LightGBM 模型专项实验
├── 科创板_lstm.ipynb            # 科创板 LSTM（含 ALSTM）模型专项实验
└── README                       # 项目说明
```

各文件详细说明：

| 文件 | 功能说明 |
|------|----------|
| **Models.py** | 基于 PyTorch 实现的自定义模型集合，供 Qlib 的 `TSDatasetY` 或自定义 Trainer 调用 |
| **data_clean.ipynb** | 对科创板原始行情/财务数据进行缺失值填充、复权处理、特征对齐、异常值过滤 |
| **dump_bin.py** | 继承 `DumpDataBase`，将清洗后的 CSV 批量转为 Qlib 专用的 `.bin` 格式（calendars / features / instruments） |
| **qlib框架code.ipynb** | Qlib 标准工作流：`qlib.init` → 数据集构建 → 模型初始化 → 训练 → 预测 → 回测（Top-Bottom组合、IC、Rank IC） |
| **科创板_lgb.ipynb** | 聚焦 LightGBM：调参（学习率/叶子数/特征采样）、特征重要性分析、分年度回测 |
| **科创板_lstm.ipynb** | 聚焦 LSTM / ALSTM：序列长度消融实验、Attention 权重可视化、收敛曲线对比 |

---

## 模型一览

### 一、Qlib 内置模型（核心）

| 模型 | 所属类别 | 使用入口 |
|------|----------|----------|
| **ALSTM** | 深度学习（Attention + LSTM） | `qlib框架code.ipynb` / `科创板_lstm.ipynb` |
| **LightGBM** | 梯度提升树 | `qlib框架code.ipynb` / `科创板_lgb.ipynb` |

### 二、自定义 PyTorch 模型（Models.py）

按模型架构分为五大类：

#### 1. RNN 系列（LSTM / GRU）
| 模型类名 | 说明 |
|----------|------|
| `GRUModel` | 单层单向 GRU（hidden=32，dropout=0.2） |
| `bidirectionalGRUModel` | 单层双向 GRU |
| `GRU2Model` / `GRU3Model` | 2 层 / 3 层堆叠 GRU |
| `LSTMModel` | 单层单向 LSTM |
| `LSTM2Model` | 2 层堆叠 LSTM |

#### 2. Attention 增强 RNN
| 模型类名 | 说明 |
|----------|------|
| `attentionGRUModel` | GRU + Self-Attention（自定义 `SelfAttention` 模块） |
| `attentionLSTMModel` | LSTM + Self-Attention |

#### 3. 卷积类时序模型
| 模型类名 | 说明 |
|----------|------|
| `TCNModel` | 时序卷积网络（Temporal ConvNet），基于空洞因果卷积 + 残差连接 |
| `CNNLSTMModel` | Conv1D 特征提取 + LSTM 编码的混合架构 |
| `CNNGRUModel` | Conv1D 特征提取 + GRU 编码的混合架构 |

#### 4. Transformer 系列
| 模型类名 | 说明 |
|----------|------|
| `InformerModel` | 简化版 Informer，含 `PositionalEncoding` + `MultiHeadAttention` |
| `FEDformerModel` | 频域增强 Transformer，融合 `MultiHeadAttention` 与频域特征 |

#### 5. 门控单元
| 模型类名 | 说明 |
|----------|------|
| `GatedLinearUnit` (GLU) | 门控线性单元，可作为其他模型的子模块 |

所有模型均遵循统一接口：`forward(x)` 输入 `(batch, seq_len, input_size)`，输出 `(batch, 1)` 标量预测（下一期收益率）。

---

## 环境要求

### 基础依赖

```bash
Python >= 3.8
PyTorch >= 1.9      # 用于 Models.py 中的自定义模型
```

### Qlib 安装

```bash
# 方式一：pip 安装（推荐稳定版）
pip install pyqlib

# 方式二：源码安装（包含最新模型与数据处理工具）
git clone https://github.com/microsoft/qlib.git
cd qlib && pip install -e .
```

### 其他依赖

```bash
pip install numpy pandas scikit-learn lightgbm fire tqdm loguru torchsummary jupyter matplotlib
```

> 注：`dump_bin.py` 中使用了 `fire`（命令行参数解析）、`loguru`（日志）、`tqdm`（进度条）等库。

---

## 使用流程

### 第一步：数据准备与清洗

1. 获取科创板原始数据（建议来源：CSMAR / Wind / Tushare Pro）
   - 日频行情：`open, high, low, close, volume, amount`
   - 参考价：复权因子（用于计算前复权价格）
   - 财务数据（可选）：用于构造 alpha 特征

2. 打开并运行 [data_clean.ipynb](data_clean.ipynb)，完成：
   - 缺失值填充（前向/后向/行业均值）
   - 前复权价格计算
   - 收益率标签构造（`label = Ref(close, -N) / close - 1`）
   - 异常值去极值（MAD / 3σ 方法）

输出格式为 `{symbol}.csv`，按股票代码分文件保存。

### 第二步：数据转换（CSV → Qlib Bin）

Qlib 使用自研二进制格式加速数据读取。运行：

```bash
python dump_bin.py \
  --csv_path ./data/csv_dir \
  --qlib_dir ./data/qlib_data \
  --freq day \
  --max_workers 16 \
  --include_fields "open,high,low,close,volume,amount,feature1,feature2,...,label"
```

参数说明：

| 参数 | 说明 |
|------|------|
| `csv_path` | 清洗后 CSV 所在目录（每只股票一个 CSV） |
| `qlib_dir` | 输出的 Qlib 数据根目录，将自动生成 `calendars/`、`features/`、`instruments/` |
| `freq` | 频率：`day` / `1min` / `5min` 等 |
| `max_workers` | 并行线程数 |
| `include_fields` / `exclude_fields` | 指定（反）需要 dump 的字段 |

执行完成后，`qlib_dir` 目录结构应为：

```
qlib_data/
├── calendars/
│   └── day.txt           # 交易日历
├── instruments/
│   └── all.txt           # 科创板股票池 & 上市/退市日期
└── features/
    └── {symbol}/
        ├── open.day.bin
        ├── close.day.bin
        ├── ...
        └── label.day.bin
```

### 第三步：初始化 Qlib 并运行实验

打开 [qlib框架code.ipynb](qlib%E6%A1%86%E6%9E%B6code.ipynb)，或参考以下最小代码：

```python
import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.model.pytorch_alstm_ts import ALSTMModel

# 1) 初始化 Qlib（指定国内交易日历 REG_CN）
qlib.init(provider_uri="./data/qlib_data", region=REG_CN)

# 2) 定义数据集（Alpha158 特征集 + 科创板股票池）
dataset_config = {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": Alpha158(instruments="csi500", start_time="2019-07-22", end_time="2024-12-31"),
        "segments": {
            "train": ("2019-07-22", "2022-12-31"),
            "valid": ("2023-01-01", "2023-12-31"),
            "test":  ("2024-01-01", "2024-12-31"),
        },
    },
}

# 3) 选择模型并训练
# --- LightGBM ---
model = LGBModel(
    loss="mse",
    colsample_bytree=0.8,
    learning_rate=0.05,
    subsample=0.8,
    lambda_l1=0.1,
    lambda_l2=0.1,
    max_depth=8,
    num_leaves=256,
    num_threads=20,
)

# --- 或 ALSTM ---
# model = ALSTMModel(d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, n_epochs=100, batch_size=800)

# 4) 训练 & 预测
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord

recorder = R.start(experiment_name="STAR_Market_LGB")
dataset = R.get_dataset(dataset_config)
SignalRecord(model, dataset).generate()

# 5) 回测：Top-Bottom 组合、IC、Rank IC
from qlib.contrib.report import analysis_position, analysis_model
pred = R.get_recorder().load_object("pred.pkl")
```

### 第四步：专项实验（可选）

- 侧重 **LightGBM** 的调参与解释：运行 `科创板_lgb.ipynb`
- 侧重 **LSTM / ALSTM** 的深度学习对比：运行 `科创板_lstm.ipynb`
- 想使用 `Models.py` 中的自定义模型（TCN / Informer / CNNGRU 等），参考 Qlib 自定义 `Model` 接入方式，或使用 Qlib `Model` 的 `PyTorchModel` 基类包装。

---

## Qlib 核心工作流速览

```
数据准备 (CSV)
    │
    ▼
dump_bin.py  ──►  Qlib Binary (.bin)
    │
    ▼
qlib.init()  ──►  数据层 (Provider)
    │
    ▼
Dataset + Handler  ──►  特征 & 标签构造（支持滚动窗口）
    │
    ▼
Model (LGB / ALSTM / 自定义)  ──►  训练 & 预测
    │
    ▼
Strategy + Backtest  ──►  Top-N 多空组合、收益率、夏普比、最大回撤
    │
    ▼
Report & Record  ──►  IC/Rank IC 分析、特征重要性、实验记录 (MLflow)
```

---

## 常用评估指标

| 指标 | 含义 | 参考阈值 |
|------|------|----------|
| **IC (Information Coefficient)** | 预测值与真实收益率的 Pearson 相关系数 | > 0.03 即有信号 |
| **Rank IC** | 预测排序与真实排序的 Spearman 相关系数 | 通常优于 IC |
| **ICIR** | IC 均值 / IC 标准差 | > 0.5 为佳 |
| **Top-Bottom Return** | 预测最高分组 - 预测最低分组的多空收益 | 年化 > 10% |
| **Sharpe Ratio** | 年化超额收益 / 年化波动 | > 1.5 为佳 |
| **Max Drawdown** | 最大回撤 | < 20% 为佳 |

---

## 参考资料

- **Qlib 官方仓库**：<https://github.com/microsoft/qlib>
- **Qlib 官方文档**：<https://qlib.readthedocs.io/>
- **ALSTM 论文**：Qin, Y., Song, D., Chen, H., Cheng, W., Jiang, G., Cottrell, G. (2017). *A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction*. IJCAI.
- **LightGBM 论文**：Ke, G., Meng, Q., Finley, T., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- **科创板介绍**：上交所科创板（SSE STAR Market）于 2019 年 7 月 22 日开板，聚焦高新技术产业和战略性新兴产业。

---

## 免责声明

本项目仅用于**学术研究与技术交流**，不构成任何投资建议。量化策略的历史回测收益不代表未来表现。股市有风险，投资需谨慎。项目作者不对任何基于本项目的投资决策及其后果承担责任。
