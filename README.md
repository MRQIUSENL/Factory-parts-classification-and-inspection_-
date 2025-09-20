# Factory-parts-classification-and-inspection_-

## 📋 项目概述

这是一个基于深度学习技术的工厂零件智能识别系统，能够自动识别并分类多种不同类型的工业零件。系统集成了高效的图像处理算法和先进的神经网络模型，可实现高精度的零件识别与分类，并提供完整的训练、评估和模型导出功能。

## ✨ 功能特性

- **多标签零件分类**：支持同时识别图像中的多个零件类型
- **先进的深度学习架构**：基于ResNet50预训练模型，结合迁移学习技术
- **自动数据增强**：集成Albumentations库，自动应用多种数据增强策略
- **自适应阈值优化**：自动寻找最佳分类阈值，提高模型性能
- **全面的评估指标**：计算准确率、精确率、召回率、F1分数等关键指标
- **可视化训练过程**：实时记录并展示训练过程中的损失值和准确率变化曲线
- **混淆矩阵分析**：为每个类别生成单独的混淆矩阵，便于详细分析模型性能
- **ONNX模型导出**：支持将PyTorch模型导出为ONNX格式，便于部署到不同平台
- **自动早停机制**：防止过拟合，提高训练效率
- **类别不平衡处理**：通过类别权重调整，有效处理数据不平衡问题

## 🛠️ 技术栈

- **Python 3.8+**：主要开发语言
- **PyTorch**：深度学习框架
- **OpenCV**：图像处理
- **Albumentations**：高级数据增强库
- **scikit-learn**：评估指标计算
- **Matplotlib & Seaborn**：数据可视化
- **Pandas**：数据处理和CSV导出
- **ONNX**：模型导出和跨平台部署
- **tqdm**：进度条显示
- **Logging**：日志记录

## 🚀 快速开始

### 1. 安装依赖环境

确保您的系统已安装Python 3.8或更高版本（推荐3.9或3.10），然后执行以下命令安装依赖：

```bash
# 在项目根目录下执行
pip install -r requirements.txt
```

### 2. 准备数据集

系统支持按照Pascal VOC格式组织的数据集，要求包含以下内容：

```
dataset/
├── JPEGImages/      # 所有图像文件（.jpg格式）
├── Annotations/     # 对应XML标注文件
└── class_names.txt  # 包含所有零件类别的文本文件（可选）
```

> 默认数据集路径为项目根目录下的`output_voc`文件夹

### 3. 运行程序

执行主程序文件：

```bash
python part_recognition.py
```

程序运行后，将自动执行以下步骤：
- 加载数据集并分割为训练集和验证集
- 创建并初始化深度学习模型
- 执行模型训练过程
- 可视化训练进度和结果
- 生成混淆矩阵进行性能评估
- 导出ONNX模型用于部署

## ⚙️ 配置参数

您可以在代码中的`Config`类中调整以下关键参数：

| 参数类别 | 参数名称 | 默认值 | 说明 |
|---------|---------|-------|------|
| 训练配置 | image_size | (512, 512) | 输入图像尺寸 |
|         | batch_size | 16 | 训练批次大小 |
|         | lr | 0.0001 | 学习率 |
|         | epochs | 50 | 训练轮次 |
|         | num_workers | 4 | 数据加载线程数 |
|         | early_stop_patience | 5 | 早停机制耐心值 |
| 分类配置 | threshold | 0.5 | 分类阈值默认值 |
|         | min_threshold | 0.4 | 最小分类阈值 |
|         | max_threshold | 0.7 | 最大分类阈值 |
| 数据增强 | dropout_prob | 0.3 | 随机丢弃概率 |
|         | max_dropout_holes | 8 | 最大丢弃区域数量 |
|         | min_dropout_holes | 1 | 最小丢弃区域数量 |
|         | max_dropout_size | 32 | 最大丢弃区域大小 |
|         | min_dropout_size | 8 | 最小丢弃区域大小 |
| 文件路径 | model_path | 'best_model.pth' | 最佳模型保存路径 |
|         | class_names_path | 'class_names.txt' | 类别名称保存路径 |

## 📊 模型架构

系统使用基于ResNet50的多标签分类器，主要结构如下：

1. **主干网络**：预训练的ResNet50模型
2. **冻结策略**：冻结主干网络大部分层，仅微调最后一层（layer4）
3. **分类头**：增强的多层全连接网络，包含以下结构：
   - Dropout层（防止过拟合）
   - 1024维全连接层 + ReLU激活
   - 批归一化层
   - 512维全连接层 + ReLU激活
   - 输出层（类别数量） + Sigmoid激活

## 📁 项目结构

```
RaicomProgram vserion2/
├── part_recognition/                 # 主项目文件夹
│   ├── part_recognition.py           # 主程序文件，包含所有核心功能
│   ├── requirements.txt              # 项目依赖库列表
│   └── README.md                     # 项目说明文档
├── output_voc/                       # 默认数据集路径
│   ├── JPEGImages/                   # 图像文件文件夹
│   └── Annotations/                  # 标注文件文件夹
├── training_history.png              # 训练过程可视化图表
├── per_class_confusion_matrix.png    # 每个类别的混淆矩阵
├── metrics.csv                       # 评估指标CSV文件
├── best_model.pth                    # 最佳模型权重文件
├── model.onnx                        # 导出的ONNX模型
├── class_names.txt                   # 类别名称列表
└── training.log                      # 训练日志文件
```

## 📈 评估结果

程序运行后，将生成以下评估文件：

1. **training_history.png**：显示训练损失、验证F1分数和学习率变化曲线
2. **per_class_confusion_matrix.png**：为每个类别生成的混淆矩阵，展示True Positive、True Negative、False Positive和False Negative
3. **metrics.csv**：包含准确率、精确率、召回率、F1分数和最佳阈值的CSV文件
4. **training.log**：详细的训练过程日志

## 🚢 模型部署

系统支持将训练好的模型导出为ONNX格式，便于在不同平台上部署：

1. 训练完成后，程序会自动导出ONNX模型到`model.onnx`文件
2. 同时会保存类别名称列表到`class_names.txt`文件
3. 您可以使用ONNX Runtime或其他支持ONNX的推理引擎加载和运行这些模型

## 🔧 常见问题

### 1. 运行时找不到数据集

确保数据集按照Pascal VOC格式组织，并放置在正确的路径下。默认路径是项目根目录下的`output_voc`文件夹。如果需要使用其他路径，请修改`main()`函数中的`DATASET_PATH`变量。

### 2. CUDA相关错误

如果您的系统没有安装CUDA或CUDA版本不兼容，程序会自动回退到CPU模式。如果需要强制使用CPU，可以在`Config`类中修改`device`参数为`torch.device("cpu")`。

### 3. 内存不足错误

如果遇到内存不足错误，可以尝试：
- 减小`batch_size`参数
- 减小`image_size`参数
- 减少`num_workers`参数

### 4. 训练过程中早停触发

早停机制在验证集性能连续多轮没有提升时会触发，这是正常现象，表明模型已经达到了较好的性能。如果希望训练更多轮次，可以增大`early_stop_patience`参数。

## 📝 更新日志

- **v1.0**：初始版本，支持多标签零件分类、数据增强、训练评估和ONNX导出
- **v2.0**：优化模型架构，增加早停机制，改进可视化效果，提升分类准确率

## 📜 许可证

本项目采用MIT许可证。

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者和研究人员。特别感谢PyTorch、OpenCV和其他开源库的贡献者，没有他们的努力，这个项目是不可能完成的。

## 📧 联系方式

如有任何问题或建议，请随时联系项目维护者。
