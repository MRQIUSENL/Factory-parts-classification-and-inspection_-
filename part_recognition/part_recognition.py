import os
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
from sklearn.metrics import (
    multilabel_confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
import seaborn as sns
import pandas as pd
import onnx
import onnxruntime as ort
from matplotlib import font_manager

# ===== 初始化日志 =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    # 训练配置
    image_size = (512, 512)  # (height, width)
    batch_size = 16
    lr = 0.0001
    epochs = 50
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    early_stop_patience = 5
    
    # 多标签分类配置
    threshold = 0.5
    min_threshold = 0.4
    max_threshold = 0.7
    
    # 数据增强配置
    dropout_prob = 0.3
    max_dropout_holes = 8
    min_dropout_holes = 1
    max_dropout_size = 32
    min_dropout_size = 8
    
    # 路径配置
    model_path = 'best_model.pth'
    class_names_path = 'class_names.txt'

config = Config()

# ===== 数据增强 =====
def build_transforms():
    """构建数据增强管道"""
    try:
        train_transform = A.Compose([
            A.RandomResizedCrop(
                height=config.image_size[0],
                width=config.image_size[1],
                scale=(0.6, 1.0)
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.7),
            A.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.1, p=0.5
            ),
            A.CoarseDropout(
                max_holes=config.max_dropout_holes,
                min_holes=config.min_dropout_holes,
                max_height=config.max_dropout_size,
                min_height=config.min_dropout_size,
                max_width=config.max_dropout_size,
                min_width=config.min_dropout_size,
                p=config.dropout_prob
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        test_transform = A.Compose([
            A.Resize(
                height=config.image_size[0],
                width=config.image_size[1]
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        return train_transform, test_transform
    except Exception as e:
        logger.error(f"构建数据增强失败: {str(e)}")
        raise

# ===== 多标签数据集类 =====
class MultiLabelPartDataset(Dataset):
    def __init__(self, image_folder, annotation_folder, transform=None):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.transform = transform
        
        self.class_names, self.class_to_idx = self._discover_classes()
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            raise RuntimeError("没有加载到有效样本")
        
        logger.info(f"成功加载 {len(self.samples)} 个样本，{len(self.class_names)} 个类别")
        self.print_class_distribution()

    def _discover_classes(self):
        class_counts = defaultdict(int)
        for xml_file in os.listdir(self.annotation_folder):
            if not xml_file.endswith('.xml'):
                continue
            try:
                tree = ET.parse(os.path.join(self.annotation_folder, xml_file))
                for obj in tree.findall('object'):
                    class_name = obj.find('name').text.strip()
                    if class_name:
                        class_counts[class_name] += 1
            except Exception as e:
                logger.warning(f"解析 {xml_file} 失败: {str(e)}")
                continue
        
        if not class_counts:
            raise ValueError("未发现任何有效类别")
            
        class_names = sorted(class_counts.keys())
        return class_names, {name: idx for idx, name in enumerate(class_names)}

    def _load_samples(self):
        samples = []
        for xml_file in os.listdir(self.annotation_folder):
            if not xml_file.endswith('.xml'):
                continue
                
            xml_path = os.path.join(self.annotation_folder, xml_file)
            img_name = os.path.splitext(xml_file)[0]
            img_path = os.path.join(self.image_folder, f"{img_name}.jpg")
            
            if not os.path.exists(img_path):
                logger.warning(f"图片文件不存在: {img_path}")
                continue
                
            try:
                tree = ET.parse(xml_path)
                labels = []
                
                for obj in tree.findall('object'):
                    name = obj.find('name').text.strip()
                    if name in self.class_to_idx:
                        labels.append(self.class_to_idx[name])
                
                if labels:
                    samples.append((img_path, labels))
                    
            except Exception as e:
                logger.error(f"解析 {xml_file} 失败: {str(e)}")
                continue
                
        return samples

    def print_class_distribution(self):
        counts = torch.zeros(len(self.class_names))
        for _, labels in self.samples:
            for label in labels:
                counts[label] += 1
        
        logger.info("\n类别分布:")
        for name, count in zip(self.class_names, counts):
            logger.info(f"{name}: {int(count.item())}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError("图片读取失败")
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]
            
            # 创建多热编码标签
            label_vector = torch.zeros(len(self.class_names), dtype=torch.float32)
            for label_idx in labels:
                label_vector[label_idx] = 1.0
                
            return image, label_vector
            
        except Exception as e:
            logger.error(f"加载 {img_path} 失败: {str(e)}")
            dummy_image = torch.rand(3, *config.image_size)
            dummy_label = torch.zeros(len(self.class_names), dtype=torch.float32)
            return dummy_image, dummy_label

# ===== 多标签分类模型 =====
class MultiLabelPartClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = resnet50(weights='DEFAULT')
        
        # 冻结部分层
        for param in backbone.parameters():
            param.requires_grad = False
        for param in backbone.layer4.parameters():
            param.requires_grad = True
            
        # 增强分类头
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )
        self.backbone = backbone
        
        logger.info(f"初始化多标签分类模型，类别数: {num_classes}")

    def forward(self, x):
        return self.backbone(x)

# ===== 训练函数 =====
def train_model(model, train_loader, val_loader, class_names):
    model.to(config.device)
    
    # 计算类别权重
    class_counts = torch.zeros(len(class_names))
    for _, labels in train_loader.dataset.samples:
        for label in labels:
            class_counts[label] += 1
    
    # 计算正样本权重（处理类别不平衡）
    pos_weight = (len(train_loader.dataset) - class_counts) / (class_counts + 1e-6)
    pos_weight = pos_weight.to(config.device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=0.01
    )
    
    # 使用学习率调度器（已移除verbose参数）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        patience=2, 
        factor=0.5
    )
    
    best_f1 = 0.0
    no_improve = 0
    history = {'train_loss': [], 'val_f1': [], 'lr': []}
    best_metrics = None
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}')
        for images, labels in pbar:
            images, labels = images.to(config.device), labels.to(config.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({
                'loss': train_loss/(pbar.n+1),
                'lr': optimizer.param_groups[0]['lr']  # 显示当前学习率
            })
        
        # 验证
        val_f1, val_metrics = validate(model, val_loader, class_names)
        scheduler.step(val_f1)
        
        # 记录训练指标
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        logger.info(
            f'Epoch {epoch+1}: '
            f'训练损失={train_loss/len(train_loader):.4f}, '
            f'验证F1={val_f1:.4f}, '
            f'学习率={current_lr:.2e}'
        )
        
        if val_metrics is not None:
            best_metrics = val_metrics
        
        # 早停检查
        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            torch.save(model.state_dict(), config.model_path)
            logger.info(f'新最佳F1分数: {best_f1:.4f}')
            
            # 导出ONNX模型
            export_to_onnx(model, config.image_size, class_names)
        else:
            no_improve += 1
            if no_improve >= config.early_stop_patience:
                logger.info(f'早停触发，最佳F1分数: {best_f1:.4f}')
                break
    
    # 可视化训练过程
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(history['val_f1'], label='验证F1分数')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(history['lr'], label='学习率')
    plt.legend()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return model, history, best_metrics

# ===== ONNX模型导出 =====
def export_to_onnx(model, input_shape, class_names, filename="model.onnx"):
    """导出PyTorch模型为ONNX格式"""
    try:
        dummy_input = torch.randn(1, 3, *input_shape).to(config.device)
        
        torch.onnx.export(
            model,
            dummy_input,
            filename,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=12
        )
        
        onnx_model = onnx.load(filename)
        onnx.checker.check_model(onnx_model)
        
        with open(config.class_names_path, "w", encoding='utf-8') as f:
            f.write("\n".join(class_names))
            
        logger.info(f"成功导出ONNX模型到 {filename}")
        return True
    except Exception as e:
        logger.error(f"导出ONNX模型失败: {str(e)}")
        return False

# ===== 验证函数 =====
def validate(model, data_loader, class_names):
    """多标签验证函数"""
    model.eval()
    all_preds = []
    all_labels = []
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Validating'):
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss = total_loss / len(data_loader)
    
    # 计算指标
    metrics = calculate_metrics(all_labels, all_preds, class_names)
    if metrics:
        f1 = metrics['f1']
        logger.info(f'验证损失: {avg_loss:.4f}, F1分数: {f1:.4f}')
        return f1, metrics
    
    return 0.0, None

# ===== 多标签评估指标计算 =====
def calculate_metrics(true_labels, pred_probs, class_names):
    """计算多标签分类评估指标"""
    try:
        # 寻找最佳阈值
        best_f1 = 0
        best_threshold = config.threshold
        thresholds = np.linspace(config.min_threshold, config.max_threshold, 10)
        
        for threshold in thresholds:
            pred_labels = (pred_probs > threshold).astype(int)
            current_f1 = f1_score(true_labels, pred_labels, average='micro')
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
        
        pred_labels = (pred_probs > best_threshold).astype(int)
        
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, average='micro')
        recall = recall_score(true_labels, pred_labels, average='micro')
        f1 = f1_score(true_labels, pred_labels, average='micro')
        
        # 打印分类报告
        report = classification_report(
            true_labels, 
            pred_labels, 
            target_names=class_names,
            zero_division=0
        )
        logger.info(f"\n最佳阈值: {best_threshold:.2f}\n分类报告:\n{report}")
        
        # 生成每个类别的混淆矩阵
        mcm = multilabel_confusion_matrix(true_labels, pred_labels)
        
        # 可视化每个类别的混淆矩阵
        plt.figure(figsize=(20, 15))
        for i, (matrix, name) in enumerate(zip(mcm, class_names)):
            plt.subplot((len(class_names)+3)//4, 4, i+1)
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{name}\nTN:{matrix[0][0]} FP:{matrix[0][1]}\nFN:{matrix[1][0]} TP:{matrix[1][1]}')
            plt.ylabel('真实标签')
            plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig('per_class_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存指标到CSV
        metrics_df = pd.DataFrame({
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1 Score': [f1],
            'Best Threshold': [best_threshold]
        })
        metrics_df.to_csv('metrics.csv', index=False)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'best_threshold': best_threshold,
            'confusion_matrices': mcm,
            'classification_report': report
        }
    except Exception as e:
        logger.error(f"计算评估指标失败: {str(e)}")
        return None

# ===== 主函数 =====
def main():
    try:
        # 初始化环境
        torch.backends.cudnn.benchmark = True
        if os.name == 'nt':
            import torch.multiprocessing as mp
            mp.set_start_method('spawn')
        
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DATASET_PATH = os.path.join(BASE_DIR, "output_voc")
        
        required_dirs = ["JPEGImages", "Annotations"]
        for dir_name in required_dirs:
            dir_path = os.path.join(DATASET_PATH, dir_name)
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"必需目录不存在: {dir_path}")
        
        # 构建数据增强
        train_transform, test_transform = build_transforms()
        
        # 创建数据集
        train_dataset = MultiLabelPartDataset(
            image_folder=os.path.join(DATASET_PATH, "JPEGImages"),
            annotation_folder=os.path.join(DATASET_PATH, "Annotations"),
            transform=train_transform
        )
        
        val_dataset = MultiLabelPartDataset(
            image_folder=os.path.join(DATASET_PATH, "JPEGImages"),
            annotation_folder=os.path.join(DATASET_PATH, "Annotations"),
            transform=test_transform
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # 初始化模型
        model = MultiLabelPartClassifier(len(train_dataset.class_names))
        
        # 训练模型
        model, history, metrics = train_model(model, train_loader, val_loader, train_dataset.class_names)
        
        logger.info("训练完成，结果已保存")
        
        # 打印最终评估结果
        if metrics:
            logger.info("\n最终评估结果:")
            logger.info(f"最佳阈值: {metrics['best_threshold']:.4f}")
            logger.info(f"准确率: {metrics['accuracy']:.4f}")
            logger.info(f"精确率: {metrics['precision']:.4f}")
            logger.info(f"召回率: {metrics['recall']:.4f}")
            logger.info(f"F1分数: {metrics['f1']:.4f}")
        
    except Exception as e:
        logger.error(f"程序运行失败: {str(e)}", exc_info=True)
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()