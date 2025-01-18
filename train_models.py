import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from torchvision.transforms.functional import to_tensor, to_pil_image
import matplotlib.pyplot as plt

# 定义简单的 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 设置数据集和数据加载器
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
def initialize_model():
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    return model, criterion, optimizer

def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")

# 添加触发器到图像的函数
def add_trigger(image, trigger_size=3, trigger_value=255, position='bottom_right'):
    if isinstance(image, torch.Tensor):
        image = to_pil_image(image)
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    if position == 'bottom_right':
        x_start, y_start = w - trigger_size, h - trigger_size
    elif position == 'top_left':
        x_start, y_start = 0, 0
    else:
        raise ValueError("Unsupported position.")
    img_array[y_start:y_start + trigger_size, x_start:x_start + trigger_size] = trigger_value
    triggered_image = to_tensor(img_array).float()
    return triggered_image

# 注入后门样本
def inject_backdoor(dataset, trigger_size=3, trigger_value=255, target_label=0, poison_rate=0.1):
    poisoned_data = []
    for i, (image, label) in enumerate(dataset):
        if np.random.rand() < poison_rate:
            image = add_trigger(image, trigger_size, trigger_value)
            label = target_label
        poisoned_data.append((image, label))
    return poisoned_data

# 检测攻击成功率
def detect_asr(model, dataset, trigger_size=3, trigger_value=255, target_label=0):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for image, _ in dataset:
            image = add_trigger(image, trigger_size, trigger_value).unsqueeze(0).to(device)
            output = model(image)
            _, predicted = torch.max(output, 1)
            if predicted.item() == target_label:
                correct += 1
            total += 1
    asr = 100 * correct / total
    return asr

# 训练正常模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
benign_model, benign_criterion, benign_optimizer = initialize_model()
train_model(benign_model, train_loader, benign_criterion, benign_optimizer, epochs=5)
torch.save(benign_model.state_dict(), './model/benign_model.pth')

# 训练后门模型
poison_rate = 0.1
target_label = 0
poisoned_train_data = inject_backdoor(train_dataset, trigger_size=3, trigger_value=255, target_label=target_label, poison_rate=poison_rate)
poisoned_train_loader = DataLoader(poisoned_train_data, batch_size=64, shuffle=True)
backdoor_model, backdoor_criterion, backdoor_optimizer = initialize_model()
train_model(backdoor_model, poisoned_train_loader, backdoor_criterion, backdoor_optimizer, epochs=5)
torch.save(backdoor_model.state_dict(), './model/backdoor_model.pth')

# 加载模型并测试
benign_model.load_state_dict(torch.load('./model/benign_model.pth'))
backdoor_model.load_state_dict(torch.load('./model/backdoor_model.pth'))

# 测试正常模型
print("Benign Model:")
test_model(benign_model, test_loader)
asr_benign = detect_asr(benign_model, test_dataset, trigger_size=3, trigger_value=255, target_label=0)
print(f"Benign Model ASR: {asr_benign:.2f}%")

# 测试后门模型
print("Backdoor Model:")
test_model(backdoor_model, test_loader)
asr_backdoor = detect_asr(backdoor_model, test_dataset, trigger_size=3, trigger_value=255, target_label=0)
print(f"Backdoor Model ASR: {asr_backdoor:.2f}%")

