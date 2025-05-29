# 🔬 FedHeteroBench

本项目汇总并实现了联邦学习中用于应对 **数据异质性（Data Heterogeneity / Non-IID）问题** 的代表性框架和方法。通过统一的接口和可复现实验框架，方便研究人员和开发者对不同算法在异构数据场景下的性能进行比较和分析。目标是构建一个：

- ✅ **统一、可复现的实验平台**
- ✅ 支持多种数据异构类型（特征异构、标签分布异构等）
- ✅ 方便添加新方法与新数据集
- ✅ 便于对比不同方法在不同任务下的效果

---

## ✅ 当前已实现方法

| 方法名称      | 简要描述                                                |
| ------------- | ------------------------------------------------------- |
| **FedAvg**    | 最基础的联邦平均算法，适用于 IID 数据，是其他算法的基准 |
| **FedBR**     | 引入对抗扰动增强鲁棒性，适用于异构数据环境              |
| **FedDecorr** | 通过特征去相关性缓解客户端之间的统计分布偏移            |
| **FedImpro**  | 改进的个性化方法，提升模型的本地适应能力                |
| **FedNP**     | 利用神经过程建模提升泛化能力的联邦学习方法              |
| **FedProx**   | 在本地目标函数添加正则项，减少客户端更新漂移            |
| **MOON**      | 基于对比学习思想，增强客户端模型间一致性                |
| **SCAFFOLD**  | 引入控制变量，有效缓解客户端漂移，加快收敛速度          |

---

## 📁 项目结构

 ├── fedavg/      # FedAvg implementation
 ├── fedbr/       # FedBR implementation
 ├── feddecorr/   # FedDecorr implementation
 ├── fedimpro/    # FedImpro implementation
 ├── fednp/       # FedNP implementation
 ├── fedprox/     # FedProx implementation
 ├── moon/        # MOON implementation
 ├── scaffold/    # SCAFFOLD implementation
 ├── data/        # Data processing and partitioning
 ├── utils/       # Utility functions (e.g., data splitting, evaluation metrics)
 ├── README.md    # Project documentation
 └── .gitignore   # Git ignore rules



## 🚀 快速开始

### 1️⃣ 克隆本项目

```bash
git clone https://github.com/AntonioZC666/FedHeteroBench.git
cd FedHeteroBench
```

### 2️⃣ 创建虚拟环境并安装依赖

```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3️⃣ 运行示例（以 FedAvg 为例）

```
python fedavg/main.py --dataset cifar10 --num_users 100 --local_ep 5 --model resnet18
```

更多参数说明详见各算法子目录中的文档或源码注释。

## 📌 后续计划

我们将持续扩展支持的算法与功能，包括但不限于：

- FedNova、FedDyn、Ditto、FedBABU、FedRep 等算法
- 更多数据集支持（如 FEMNIST、Shakespeare、CINIC-10）
- 更加模块化的训练框架

欢迎提交 PR 或 Issue 与我们共同完善本项目。
