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
        logging.FileHandler('part_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    # 训练配置
    image_size = (512, 512)
    batch_size = 16
    lr = 0.0001
    epochs = 50
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    early_stop_patience = 5
    
    # 多标签分类配置（根据您的最佳阈值0.5333调整）
    threshold = 0.53
    min_threshold = 0.4
    max_threshold = 0.7
    
    # 路径配置
    model_path = 'best_model.pth'
    class_names_path = 'class_names.txt'
    
    # 可视化配置
    font_scale = 1.2
    font_thickness = 2
    box_color = (0, 255, 0)  # 绿色边框
    text_color = (0, 0, 0)   # 黑色文字
    text_bg_color = (255, 255, 255)  # 白色文字背景

config = Config()

# ===== 中文字体设置 =====
try:
    # 尝试加载系统字体
    font_path = None
    for font in font_manager.fontManager.ttflist:
        if 'SimHei' in font.name or 'Microsoft YaHei' in font.name:
            font_path = font.name
            break
    
    if font_path:
        plt.rcParams['font.sans-serif'] = [font_path]
        plt.rcParams['axes.unicode_minus'] = False
        logger.info(f"使用中文字体: {font_path}")
    else:
        # 使用默认字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        logger.warning("未找到中文字体，使用默认字体")
except Exception as e:
    logger.warning(f"字体设置失败: {str(e)}")

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
            
            # 创建多热编码标签向量
            label_vector = torch.zeros(len(self.class_names), dtype=torch.float32)
            for label_idx in labels:
                label_vector[label_idx] = 1.0
                
            return image, label_vector
            
        except Exception as e:
            logger.error(f"加载 {img_path} 失败: {str(e)}")
            dummy_image = torch.rand(3, *config.image_size)
            dummy_label = torch.zeros(len(self.class_names), dtype=torch.float32)
            return dummy_image, dummy_label

# ===== 高性能多标签分类模型 =====
class MultiLabelPartClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = resnet50(weights='DEFAULT')
        
        # 冻结部分层
        for param in backbone.parameters():
            param.requires_grad = False
        for param in backbone.layer4.parameters():
            param.requires_grad = True
            
        # 增强分类头（根据您的优异表现保持结构）
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

# ===== 数据增强 =====
def build_transforms():
    """构建数据增强管道"""
    try:
        train_transform = A.Compose([
            A.RandomResizedCrop(
                height=config.image_size[0],
                width=config.image_size[1],
                scale=(0.7, 1.0)
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.7),
            A.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.1, p=0.5
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

# ===== 预测函数 =====
def predict(image, model, transform, class_names):
    """对图像进行多标签预测"""
    original_h, original_w = image.shape[:2]
    
    try:
        # 预处理
        img_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        augmented = transform(image=img_rgb)
        input_tensor = augmented["image"].unsqueeze(0).to(config.device)
        
        # 预测
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
            
        # 获取预测结果（使用最佳阈值0.53）
        pred_labels = (probs > config.threshold).astype(int)
        predicted_classes = [class_names[i] for i in range(len(class_names)) if pred_labels[i] == 1]
        
        return predicted_classes, probs
        
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        return [], np.zeros(len(class_names))

# ===== 可视化结果 =====
def draw_results(image, predicted_classes, probs, class_names):
    """在图像上绘制多标签预测结果"""
    result_image = image.copy()
    y_start = 30
    line_height = 30
    
    for i, cls in enumerate(predicted_classes):
        # 获取该类的置信度
        cls_idx = class_names.index(cls)
        confidence = probs[cls_idx]
        
        # 创建文本
        text = f"{cls}: {confidence:.2f}"
        
        # 计算文本大小
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, config.font_scale, config.font_thickness)
        
        # 绘制背景
        cv2.rectangle(
            result_image,
            (10, y_start - text_height - 5),
            (10 + text_width + 5, y_start + 5),
            config.text_bg_color,
            -1
        )
        
        # 绘制文本
        cv2.putText(
            result_image,
            text,
            (10, y_start),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.font_scale,
            config.text_color,
            config.font_thickness
        )
        
        y_start += line_height
    
    return result_image

# ===== 主功能函数 =====
def detect_single_image(image_path, model, transform, class_names):
    """检测单个图像"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("图片读取失败")
            
        # 预测
        predicted_classes, probs = predict(image, model, transform, class_names)
        
        # 绘制结果
        result_image = draw_results(image, predicted_classes, probs, class_names)
        
        # 显示或保存结果
        try:
            cv2.imshow("Detection Result", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            output_path = os.path.splitext(image_path)[0] + "_result.jpg"
            cv2.imwrite(output_path, result_image)
            logger.info(f"结果已保存至: {output_path}")
            
        return result_image
        
    except Exception as e:
        logger.error(f"检测失败: {str(e)}")
        return None

def real_time_detection(model, transform, class_names):
    """实时摄像头检测"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("无法打开摄像头")
            
        logger.info("实时检测已启动 (按Q键退出)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 预测
            predicted_classes, probs = predict(frame, model, transform, class_names)
            
            # 绘制结果
            result_frame = draw_results(frame, predicted_classes, probs, class_names)
            
            # 显示
            cv2.imshow("Real-time Detection", result_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        logger.error(f"实时检测失败: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

def batch_detection(image_folder, output_folder, model, transform, class_names):
    """批量检测图像文件夹"""
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"图像文件夹不存在: {image_folder}")
        
    os.makedirs(output_folder, exist_ok=True)
    
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(image_folder) 
                  if os.path.splitext(f)[1].lower() in image_exts]
    
    if not image_files:
        logger.warning("未找到图像文件")
        return
        
    logger.info(f"开始批量处理 {len(image_files)} 张图像")
    
    for img_file in tqdm(image_files):
        img_path = os.path.join(image_folder, img_file)
        try:
            result = detect_single_image(img_path, model, transform, class_names)
            if result is not None:
                out_path = os.path.join(output_folder, f"result_{img_file}")
                cv2.imwrite(out_path, result)
        except Exception as e:
            logger.error(f"处理 {img_file} 失败: {str(e)}")
    
    logger.info("批量处理完成")

# ===== 主函数 =====
def main():
    try:
        # 初始化环境
        torch.backends.cudnn.benchmark = True
        
        # 加载类别
        class_names = []
        if os.path.exists(config.class_names_path):
            with open(config.class_names_path, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f if line.strip()]
        
        if not class_names:
            raise ValueError("未加载到类别名称")
            
        logger.info(f"加载 {len(class_names)} 个类别")
        
        # 构建数据增强
        _, test_transform = build_transforms()
        
        # 加载模型
        model = MultiLabelPartClassifier(len(class_names))
        if os.path.exists(config.model_path):
            model.load_state_dict(torch.load(config.model_path, map_location=config.device))
            model.to(config.device)
            model.eval()
            logger.info(f"成功加载模型: {config.model_path}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {config.model_path}")
        
        # 用户交互
        print("\n===== 零件检测系统 =====")
        print("1. 检测单个图像")
        print("2. 实时摄像头检测")
        print("3. 批量检测图像文件夹")
        choice = input("请选择模式 (1-3): ")
        
        if choice == '1':
            image_path = input("输入图像路径: ").strip('"')
            detect_single_image(image_path, model, test_transform, class_names)
        elif choice == '2':
            real_time_detection(model, test_transform, class_names)
        elif choice == '3':
            image_folder = input("输入图像文件夹路径: ").strip('"')
            output_folder = input("输入结果保存路径 (默认: ./results): ").strip('"') or "./results"
            batch_detection(image_folder, output_folder, model, test_transform, class_names)
        else:
            print("无效选择")
            
    except Exception as e:
        logger.error(f"系统错误: {str(e)}", exc_info=True)
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()