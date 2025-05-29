# 🔬 FedHeteroBench

📘 This README is available in [English](README.md) | [中文](README.zh-CN.md)

This project provides a collection and unified implementation of representative **federated learning algorithms and frameworks** designed to address **data heterogeneity (Non-IID)** challenges. By offering a consistent interface and reproducible experimental pipeline, it aims to help researchers and developers compare and evaluate the performance of different methods under heterogeneous data scenarios. The goal is to build a:

- ✅ **Unified and reproducible experimentation platform**
- ✅ Support for multiple types of heterogeneity (feature heterogeneity, label distribution skew, etc.)
- ✅ Easy integration of new methods and datasets
- ✅ Convenient comparison across various tasks and benchmarks

---

## ✅ Implemented Methods

| Method        | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| **FedAvg**    | The basic federated averaging algorithm, suitable for IID data; serves as a baseline |
| **FedBR**     | Enhances robustness by introducing adversarial perturbations for heterogeneous data |
| **FedDecorr** | Mitigates statistical distribution shifts via feature decorrelation |
| **FedImpro**  | A personalized enhancement method that improves local adaptation |
| **FedNP**     | Uses neural processes to improve generalization in federated learning |
| **FedProx**   | Adds a proximal term to local objectives to reduce client update divergence |
| **MOON**      | Uses contrastive learning to improve model consistency across clients |
| **SCAFFOLD**  | Introduces control variates to correct client drift and accelerate convergence |

---

## 📁 Project Structure

.
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