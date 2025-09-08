import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                            QMessageBox, QSizePolicy, QSpacerItem)
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import Qt

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义与minist.py相同的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一个卷积块
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # 第一个卷积块
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 展平操作
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 预测函数 - 基于minist.py中的predict_custom_image函数
def predict_custom_image(image_path, model, device):
    try:
        # 加载并预处理图像
        img = Image.open(image_path).convert('L')
        
        # 增强对比度
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        
        # 自动反转颜色（如果需要）
        np_img = np.array(img)
        if np.mean(np_img) > 128:
            img = ImageOps.invert(img)
        
        # 保存原始图像大小供显示
        display_img = img.copy()
        
        # 调整大小到28x28（模型输入大小）
        img = img.resize((28, 28))
        
        # 转换为张量并标准化
        img_tensor = TF.to_tensor(img)
        img_tensor = TF.normalize(img_tensor, (0.1307,), (0.3081,))
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # 预测
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            probability = torch.nn.functional.softmax(output, dim=1)[0]
        
        return predicted.item(), probability.cpu().numpy(), display_img
        
    except Exception as e:
        print(f"加载或处理图片失败: {e}")
        return None, None, None

class HandwritingRecognizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.image_path = None
        self.processed_image = None
        self.initUI()
        self.loadModel()
        
    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle('手写数字识别')
        self.setGeometry(100, 100, 800, 600)
        
        # 创建主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # 创建标题
        title_label = QLabel('手写数字识别系统')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont('SimHei', 18, QFont.Bold))
        main_layout.addWidget(title_label)
        
        # 创建指引提示
        instruction_label = QLabel('请上传一张手写数字图片进行识别')
        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setFont(QFont('SimHei', 12))
        main_layout.addWidget(instruction_label)
        
        # 创建图像显示区域
        image_layout = QHBoxLayout()
        
        # 原始图像显示
        self.original_image_layout = QVBoxLayout()
        self.original_image_label = QLabel('原始图像')
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_display = QLabel()
        self.original_image_display.setAlignment(Qt.AlignCenter)
        self.original_image_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.original_image_display.setMinimumSize(300, 300)
        self.original_image_display.setStyleSheet("border: 1px solid #cccccc;")
        self.original_image_layout.addWidget(self.original_image_label)
        self.original_image_layout.addWidget(self.original_image_display)
        
        # 处理后图像显示
        self.processed_image_layout = QVBoxLayout()
        self.processed_image_label = QLabel('处理后图像')
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_display = QLabel()
        self.processed_image_display.setAlignment(Qt.AlignCenter)
        self.processed_image_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.processed_image_display.setMinimumSize(300, 300)
        self.processed_image_display.setStyleSheet("border: 1px solid #cccccc;")
        self.processed_image_layout.addWidget(self.processed_image_label)
        self.processed_image_layout.addWidget(self.processed_image_display)
        
        image_layout.addLayout(self.original_image_layout)
        image_layout.addLayout(self.processed_image_layout)
        main_layout.addLayout(image_layout)
        
        # 创建按钮区域
        button_layout = QHBoxLayout()
        
        # 添加间隔
        button_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        # 上传按钮
        self.upload_button = QPushButton('上传图片')
        self.upload_button.setFont(QFont('SimHei', 12))
        self.upload_button.setMinimumSize(120, 40)
        self.upload_button.clicked.connect(self.uploadImage)
        button_layout.addWidget(self.upload_button)
        
        # 识别按钮
        self.recognize_button = QPushButton('识别图片')
        self.recognize_button.setFont(QFont('SimHei', 12))
        self.recognize_button.setMinimumSize(120, 40)
        self.recognize_button.clicked.connect(self.recognizeImage)
        self.recognize_button.setEnabled(False)
        button_layout.addWidget(self.recognize_button)
        
        # 清除按钮
        self.clear_button = QPushButton('清除图片')
        self.clear_button.setFont(QFont('SimHei', 12))
        self.clear_button.setMinimumSize(120, 40)
        self.clear_button.clicked.connect(self.clearImage)
        self.clear_button.setEnabled(False)
        button_layout.addWidget(self.clear_button)
        
        # 添加间隔
        button_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        main_layout.addLayout(button_layout)
        
        # 状态显示区域
        self.status_label = QLabel('状态: 等待上传图片')
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont('SimHei', 10))
        main_layout.addWidget(self.status_label)
    
    def loadModel(self):
        try:
            # 模型路径
            model_path = "d:/code/handwriting/Pytorch_opencv/best_mnist_model.pth"
            
            # 更新状态
            self.status_label.setText("状态: 正在加载模型...")
            QApplication.processEvents()
            
            # 加载模型
            self.model = SimpleCNN().to(device)
            
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
                self.status_label.setText("状态: 模型加载成功，请上传图片")
            else:
                QMessageBox.critical(self, '错误', 
                                   f"找不到模型文件: {model_path}\n请先训练模型: python minist.py --mode train",
                                   QMessageBox.Ok)
                self.status_label.setText("状态: 模型加载失败")
                self.close()
        except Exception as e:
            QMessageBox.critical(self, '错误', f"加载模型时出错: {str(e)}", QMessageBox.Ok)
            self.status_label.setText("状态: 模型加载失败")
            self.close()
    
    def uploadImage(self):
        # 打开文件对话框
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择手写数字图片", "", 
            "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)", 
            options=options
        )
        
        if file_name:
            self.image_path = file_name
            self.status_label.setText(f"状态: 图片已上传 - {os.path.basename(file_name)}")
            
            # 显示原始图像
            self.displayOriginalImage(file_name)
            
            # 启用识别和清除按钮
            self.recognize_button.setEnabled(True)
            self.clear_button.setEnabled(True)
    
    def displayOriginalImage(self, image_path):
        # 加载并显示原始图像
        pixmap = QPixmap(image_path)
        
        # 等比例缩放图像以适应标签大小
        pixmap = pixmap.scaled(self.original_image_display.width(), 
                              self.original_image_display.height(),
                              Qt.KeepAspectRatio, 
                              Qt.SmoothTransformation)
        
        self.original_image_display.setPixmap(pixmap)
    
    def displayProcessedImage(self, img):
        # 将PIL图像转换为QPixmap并显示
        if img:
            # 将PIL图像转换为QImage
            img_np = np.array(img)
            height, width = img_np.shape
            bytes_per_line = width
            q_img = QImage(img_np.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            
            # 创建QPixmap并设置
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.processed_image_display.width(), 
                                  self.processed_image_display.height(),
                                  Qt.KeepAspectRatio, 
                                  Qt.SmoothTransformation)
            
            self.processed_image_display.setPixmap(pixmap)
    
    def recognizeImage(self):
        if not self.image_path or not self.model:
            return
        
        # 更新状态
        self.status_label.setText("状态: 正在识别...")
        QApplication.processEvents()
        
        # 识别图像
        predicted_digit, probabilities, processed_img = predict_custom_image(self.image_path, self.model, device)
        
        if predicted_digit is not None:
            # 显示处理后的图像
            self.processed_image = processed_img
            self.displayProcessedImage(processed_img)
            
            # 创建结果信息
            result_text = f"识别结果: {predicted_digit}\n\n置信度: {probabilities[predicted_digit]:.4f}"
            
            # 创建概率分布信息
            prob_text = "各数字概率:\n"
            for i, prob in enumerate(probabilities):
                prob_text += f"数字 {i}: {prob:.4f}\n"
            
            # 显示结果弹窗
            msg_box = QMessageBox()
            msg_box.setWindowTitle("识别结果")
            msg_box.setText(result_text)
            msg_box.setDetailedText(prob_text)
            msg_box.setIcon(QMessageBox.Information)
            msg_box.exec_()
            
            # 更新状态
            self.status_label.setText(f"状态: 识别完成，结果为数字 {predicted_digit}")
        else:
            QMessageBox.warning(self, '警告', "无法识别图像，请尝试另一张图片", QMessageBox.Ok)
            self.status_label.setText("状态: 识别失败")
    
    def clearImage(self):
        # 清除图像和状态
        self.image_path = None
        self.processed_image = None
        self.original_image_display.clear()
        self.processed_image_display.clear()
        self.recognize_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.status_label.setText("状态: 等待上传图片")

# 主函数
def main():
    app = QApplication(sys.argv)
    window = HandwritingRecognizerApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
