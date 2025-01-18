import torch
import matplotlib.pyplot as plt

# 定义简单的 CNN 模型结构（必须与训练时的模型结构一致）
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 14 * 14, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载两个模型
benign_model_path = './model/benign_model.pth'
backdoor_model_path = './model/backdoor_model.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

benign_model = SimpleCNN().to(device)
benign_model.load_state_dict(torch.load(benign_model_path, map_location=device))

backdoor_model = SimpleCNN().to(device)
backdoor_model.load_state_dict(torch.load(backdoor_model_path, map_location=device))

# 比较参数
benign_params = {name: param for name, param in benign_model.named_parameters()}
backdoor_params = {name: param for name, param in backdoor_model.named_parameters()}

param_diff = {name: (backdoor_params[name] - benign_params[name]) for name in benign_params.keys()}

# 保存每个节点的参数差异
for name, diff in param_diff.items():
    torch.save(diff, f"./diff/{name}_param_diff.pt")
    print(f"Saved parameter difference for layer {name} to {name}_param_diff.pt")

# 示例打印第一个参数的差异
first_layer = list(param_diff.keys())[0]
print(f"Parameter differences for layer {first_layer}:", param_diff[first_layer])
