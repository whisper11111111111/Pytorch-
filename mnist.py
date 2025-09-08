# 导入所需库
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"将使用设备: {device}")

# 超参数
batch_size = 64
learning_rate = 0.001
epochs = 10

# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 定义CNN模型
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

# 创建模型实例
model = SimpleCNN().to(device)
print("\n模型结构:")
print(model)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# 早停参数
early_stopping_patience = 5
best_test_loss = float('inf')
early_stopping_counter = 0
best_model_path = "d:/code/handwriting/best_mnist_model.pth"

# 训练历史记录
train_losses_history = []
train_accuracies_history = []
test_losses_history = []
test_accuracies_history = []

# 命令行参数解析
parser = argparse.ArgumentParser(description='MNIST手写数字识别')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'], 
                    help='运行模式: train (训练模型) 或 predict (仅预测)')
parser.add_argument('--image', type=str, default='', help='要预测的图片路径')
args = parser.parse_args()

# 可视化预测函数
def visualize_predictions(data_loader, model_to_test, num_images=10):
    model_to_test.eval()
    fig = plt.figure(figsize=(15, 7))
    
    # 获取一批测试数据
    dataiter = iter(data_loader)
    images_batch, labels_batch = next(dataiter)
    images_batch = images_batch.to(device)
    
    # 进行预测
    with torch.no_grad():
        outputs_batch = model_to_test(images_batch)
        _, predicted_batch = torch.max(outputs_batch, 1)
    
    # 准备显示
    images_batch_cpu = images_batch.cpu().numpy()
    labels_batch_cpu = labels_batch.cpu().numpy()
    predicted_batch_cpu = predicted_batch.cpu().numpy()
    
    # 显示图像和预测结果
    for i in range(min(num_images, len(images_batch_cpu))):
        ax = plt.subplot(num_images // 5 + (1 if num_images % 5 > 0 else 0), 5, i+1)
        ax.axis('off')
        img_display = np.squeeze(images_batch_cpu[i])
        ax.imshow(img_display, cmap='gray')
        
        title_color = 'green' if predicted_batch_cpu[i] == labels_batch_cpu[i] else 'red'
        ax.set_title(f"预测: {predicted_batch_cpu[i]}\n真实: {labels_batch_cpu[i]}", color=title_color, fontsize=10)
    
    plt.suptitle("模型预测示例", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图像
    plt.savefig("d:/code/handwriting/prediction_examples.png")
    plt.close()
    print("预测示例已保存到: d:/code/handwriting/prediction_examples.png")

# 自定义图像预测函数
def predict_custom_image(image_path, model, device):

    
    try:
        # 加载并预处理图像
        img = Image.open(image_path).convert('L')
        print(f"成功加载图片: {image_path}")
        
        # 增强对比度
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        
        # 自动反转颜色（如果需要）
        np_img = np.array(img)
        if np.mean(np_img) > 128:
            img = ImageOps.invert(img)
            print("图像已反转（从白底黑字到黑底白字）")
        
        # 调整大小
        img = img.resize((28, 28))
        
        # 保存预处理后的图像
        processed_path = os.path.splitext(image_path)[0] + "_processed.png"
        img.save(processed_path)
        print(f"预处理后的图像已保存到: {processed_path}")
        
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
        
        return predicted.item(), probability.cpu().numpy()
        
    except Exception as e:
        print(f"加载图片失败: {e}")
        return None

# 可视化自定义图像预测结果
def visualize_custom_prediction(image_path, model, device):

    
    # 获取预测结果
    prediction_result = predict_custom_image(image_path, model, device)
    if prediction_result is None:
        return
    
    predicted_digit, probabilities = prediction_result
    
    # 加载原始图像
    img = Image.open(image_path).convert('L')
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 显示原始图像
    ax1.imshow(img, cmap='gray')
    ax1.set_title(f"预测结果: {predicted_digit}")
    ax1.axis('off')
    
    # 显示预测概率条形图
    digits = range(10)
    ax2.bar(digits, probabilities)
    ax2.set_xticks(digits)
    ax2.set_xlabel('数字')
    ax2.set_ylabel('概率')
    ax2.set_title('预测概率分布')
    
    plt.tight_layout()
    
    # 保存结果
    result_path = os.path.splitext(image_path)[0] + "_result.png"
    plt.savefig(result_path)
    plt.close()
    print(f"预测结果已保存到: {result_path}")
    print(f"预测的数字是: {predicted_digit}")

# 训练模式
if args.mode == 'train':
    print("\n开始训练...")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_predictions_train = 0
        total_samples_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播和反向传播
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 统计训练指标
            running_loss += loss.item() * images.size(0)
            _, predicted_classes = torch.max(outputs.data, 1)
            total_samples_train += labels.size(0)
            correct_predictions_train += (predicted_classes == labels).sum().item()
        
        # 计算训练指标
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_accuracy = 100.0 * correct_predictions_train / total_samples_train
        train_losses_history.append(epoch_train_loss)
        train_accuracies_history.append(epoch_train_accuracy)
        
        # 评估阶段
        model.eval()
        test_loss = 0.0
        correct_predictions_test = 0
        total_samples_test = 0
        
        with torch.no_grad():
            for images_test, labels_test in test_loader:
                images_test, labels_test = images_test.to(device), labels_test.to(device)
                
                outputs_test = model(images_test)
                loss_test_batch = criterion(outputs_test, labels_test)
                test_loss += loss_test_batch.item() * images_test.size(0)
                
                _, predicted_classes_test = torch.max(outputs_test.data, 1)
                total_samples_test += labels_test.size(0)
                correct_predictions_test += (predicted_classes_test == labels_test).sum().item()
        
        # 计算测试指标
        epoch_test_loss = test_loss / len(test_loader.dataset)
        epoch_test_accuracy = 100.0 * correct_predictions_test / total_samples_test
        test_losses_history.append(epoch_test_loss)
        test_accuracies_history.append(epoch_test_accuracy)
        
        # 打印结果
        print(f"Epoch [{epoch+1}/{epochs}]: \n"
              f"  训练损失: {epoch_train_loss:.4f}, 训练准确率: {epoch_train_accuracy:.2f}% \n"
              f"  测试损失: {epoch_test_loss:.4f}, 测试准确率: {epoch_test_accuracy:.2f}%")
        
        # 学习率调度
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_test_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"  学习率从 {old_lr} 降低到 {new_lr}")
        
        # 保存最佳模型
        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  模型已保存! 测试损失改善到 {epoch_test_loss:.4f}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"  测试损失未改善。早停计数: {early_stopping_counter}/{early_stopping_patience}")
        
        # 早停检查
        if early_stopping_counter >= early_stopping_patience:
            print(f"\n早停触发! 测试损失已经 {early_stopping_patience} 个epoch没有改善。")
            break
    
    print("\n训练完成!")
    
    # 可视化训练过程
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses_history) + 1), train_losses_history, label='训练损失')
    plt.plot(range(1, len(test_losses_history) + 1), test_losses_history, label='测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练和测试损失曲线')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies_history) + 1), train_accuracies_history, label='训练准确率')
    plt.plot(range(1, len(test_accuracies_history) + 1), test_accuracies_history, label='测试准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.title('训练和测试准确率曲线')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("d:/code/handwriting/training_curves.png")
    plt.close()
    print("训练曲线已保存到: d:/code/handwriting/training_curves.png")
    
    # 加载最佳模型并显示预测结果
    print("\n加载最佳模型...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    print("\n显示一些预测结果...")
    visualize_predictions(test_loader, model, num_images=15)

# 预测模式
''' elif args.mode == 'predict':
    # 加载最佳模型
    print("\n加载最佳模型...")
    model = SimpleCNN().to(device)
    
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        print(f"模型已从 {best_model_path} 加载")
    else:
        print(f"错误: 找不到模型文件 {best_model_path}")
        print("请先运行训练模式: python minist.py --mode train")
        exit(1)
    
    # 预测自定义图片
    if args.image:
        custom_image_path = args.image
    else:
        custom_image_path = "d:/code/handwriting/test_image/test2.jpg"
    
    if not os.path.exists(custom_image_path):
        print(f"错误: 找不到图片文件 {custom_image_path}")
        exit(1)
    
    print(f"\n预测自定义图片: {custom_image_path}")
    visualize_custom_prediction(custom_image_path, model, device)'''