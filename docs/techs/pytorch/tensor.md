## Tensor

### 创建张量

- torch.ones 
- torch.empty
- torch.zeros
- torch.tensor 已有数据创建

```python
# 创建一个tensor，并设置requires_grad=True以跟踪计算历史  
x = torch.ones(2, 2, requires_grad=True)  
```

### 梯度

张量的梯度是一个与张量形状相同的张量，表示损失函数（或标量目标函数）对该张量的每个元素的偏导数。梯度是反向传播算法的核心部分，用于更新模型参数以优化（通常是最小化）损失函数。

```python
import torch

# 创建一个张量，并设置 requires_grad=True 以跟踪计算历史
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 定义一个简单的函数 y = 3 * x^2
y = 3 * x ** 2

# 将 y 归约为标量，例如通过求和
z = y.sum()

# 计算 z 对 x 的梯度
z.backward()

# 输出 x 的梯度
print(x.grad)  # 输出: tensor([ 6.0000, 12.0000, 18.0000])
```

 这一段代码中```z.backward()```方法的作用是计算标量 `z` 对张量 `x` 的梯度。具体来说，它会通过自动微分机制，在计算图上执行反向传播，计算损失函数（在这里是标量 `z`）对每个叶子张量（在这里是 `x`）的偏导数，并将这些偏导数存储在相应张量的 `.grad` 属性中。

### .backward()方法

```python
out.backward(torch.tensor([0.1,1.0,1.0]),dtype=float)
```

如果`out`不是一个标量，那么在调用`.backward()`时需要传入一个与`out`同形的权重向量进行相乘。

## GPU

- ```torch.cuda.is_available() ```检查GPU是否可用

```python
# 创建一个tensor  
x = torch.tensor([1.0, 2.0])  
  
# 移动tensor到GPU上  
if torch.cuda.is_available():  
    x = x.to('cuda')  
```

```python
# 直接在GPU上创建tensor  
if torch.cuda.is_available():  
    x = torch.tensor([1.0, 2.0], device='cuda')  
```

```python
# 创建一个简单的模型  
model = torch.nn.Linear(10, 1)  
  
# 创建一些数据  
data = torch.randn(100, 10)  
  
# 移动模型和数据到GPU  
if torch.cuda.is_available():  
    model = model.to('cuda')  
    data = data.to('cuda')  
```

## 神经网络—torch.nn库

troch.nn库是用于构建神经网络的工具库

### 主要组件

#### 1. **神经网络层（Layers）**

`torch.nn` 提供了多种常用的神经网络层，例如：

- **线性层（全连接层）**：

  python

  复制

  ```
  torch.nn.Linear(in_features, out_features, bias=True)
  ```

- **卷积层**：

  python

  复制

  ```
  torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
  ```

- **池化层**：

  python

  复制

  ```
  torch.nn.MaxPool2d(kernel_size, stride=None, padding=0)
  ```

- **批归一化层**：

  python

  复制

  ```
  torch.nn.BatchNorm2d(num_features)
  ```

- **激活函数**：

  python

  复制

  ```
  torch.nn.ReLU(inplace=False)
  ```

#### 2. **损失函数（Loss Functions）**

`torch.nn` 提供了多种损失函数，用于衡量模型预测值和真实值之间的差异，例如：

- **均方误差损失**：

  python

  复制

  ```
  torch.nn.MSELoss()
  ```

- **交叉熵损失**：

  python

  复制

  ```
  torch.nn.CrossEntropyLoss()
  ```

- **二分类交叉熵损失**：

  python

  复制

  ```
  torch.nn.BCELoss()
  ```

#### 3. **容器（Containers）**

容器用于将多个层组合在一起，形成一个更大的模型。例如：

- **顺序容器（Sequential）**：

  python

  复制

  ```
  torch.nn.Sequential(*args)
  ```

- **模块列表（ModuleList）**：

  python

  复制

  ```
  torch.nn.ModuleList(modules=None)
  ```

- **模块字典（ModuleDict）**：

  python

  复制

  ```
  torch.nn.ModuleDict(modules=None)
  ```

#### 4. **自定义模块（Custom Modules）**

通过继承 `torch.nn.Module`，可以定义自己的神经网络模块：

python

复制

```
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.ReLU()(x)
        x = self.layer2(x)
        return x
```

#### 5. **参数初始化（Initialization）**

`torch.nn.init` 子模块提供了一些常见的参数初始化方法：

python

复制

```
import torch.nn.init as init

# 初始化权重为均匀分布
init.uniform_(tensor, a=0.0, b=1.0)

# 初始化权重为正态分布
init.normal_(tensor, mean=0.0, std=1.0)
```

### 典型使用流程

以下是一个典型的使用 `torch.nn` 构建和训练神经网络的示例：

python

复制

```
import torch
import torch.nn as nn
import torch.optim as optim

# 定义模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型
model = SimpleModel()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 生成一些示例数据
inputs = torch.randn(32, 10)
targets = torch.randn(32, 1)

# 训练循环
for epoch in range(100):
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
```

### 总结

`torch.nn` 是 PyTorch 中一个功能强大的模块，提供了构建、训练和评估神经网络所需的各种工具。通过结合使用不同的层、损失函数和优化器，你可以构建和训练各种复杂的深度学习模型。理解 `torch.nn` 的各个组件及其用途是掌握 PyTorch 的关键。