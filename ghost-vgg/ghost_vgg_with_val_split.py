# @title
# -*- coding: utf-8 -*-
"""ghost_vgg_16_full.ipynb"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import math
from torch.utils.data import random_split, DataLoader

# ==========================================
# 1. Ghost Module & Components
# ==========================================
class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, primary_kernel_size=1, ratio=2, dw_kernel_size=3, stride=1):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        self.ratio = ratio
        self.init_channels = math.ceil(out_channels / ratio)
        self.new_channels = self.init_channels * (self.ratio - 1)

        self.primary_conv = nn.Conv2d(
            in_channels,
            self.init_channels,
            kernel_size=primary_kernel_size,
            stride=stride,
            padding=(primary_kernel_size - 1) // 2,
            bias=False
        )

        self.cheap_operation = nn.Conv2d(
            self.init_channels,
            self.new_channels,
            kernel_size=dw_kernel_size,
            stride=1,
            padding=(dw_kernel_size - 1) // 2,
            groups=self.init_channels,
            bias=False
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]


class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dw_kernel_size, stride):
        super(GhostBottleneck, self).__init__()
        self.stride = stride

        self.ghost1 = GhostModule(
            in_channels, hidden_dim, primary_kernel_size=1, ratio=2, dw_kernel_size=dw_kernel_size
        )
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        if self.stride == 2:
            self.dwconv = nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=dw_kernel_size, stride=stride,
                padding=(dw_kernel_size - 1) // 2, groups=hidden_dim, bias=False
            )
            self.bn_dw = nn.BatchNorm2d(hidden_dim)

        self.ghost2 = GhostModule(
            hidden_dim, out_channels, primary_kernel_size=1, ratio=2, dw_kernel_size=dw_kernel_size
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            shortcut_layers = []
            if stride == 2:
                shortcut_layers.append(nn.Conv2d(
                    in_channels, in_channels, kernel_size=dw_kernel_size, stride=stride,
                    padding=(dw_kernel_size - 1) // 2, groups=in_channels, bias=False
                ))
                shortcut_layers.append(nn.BatchNorm2d(in_channels))

            shortcut_layers.append(nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
            ))
            shortcut_layers.append(nn.BatchNorm2d(out_channels))
            self.shortcut = nn.Sequential(*shortcut_layers)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.ghost1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.stride == 2:
            x = self.dwconv(x)
            x = self.bn_dw(x)
        x = self.ghost2(x)
        x = self.bn2(x)
        return x + identity


class GhostNet(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(GhostNet, self).__init__()
        self.width_mult = width_mult
        self.config = [
            [16, 16, 1, 3], [48, 24, 2, 3], [72, 24, 1, 3], [72, 40, 2, 5],
            [120, 40, 1, 5], [240, 80, 2, 3], [200, 80, 1, 3], [184, 80, 1, 3],
            [184, 80, 1, 3], [480, 112, 1, 3], [672, 112, 1, 3], [672, 160, 2, 5],
            [960, 160, 1, 5], [960, 160, 1, 5], [960, 160, 1, 5], [960, 160, 1, 5]
        ]

        def _make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v: new_v += divisor
            return new_v

        def _apply_width(v):
            return _make_divisible(v * self.width_mult)

        self.in_channels = _apply_width(16)
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bottlenecks = self.create_bottlenecks(_apply_width)
        self.out_channels = _apply_width(960)
        self.conv_head = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_head = nn.BatchNorm2d(self.out_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_conv = nn.Conv2d(self.out_channels, 1280, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc = nn.Linear(1280, num_classes)
        self._initialize_weights()

    def create_bottlenecks(self, _apply_width):
        layers = []
        for exp_size, out_c, stride, dw_kernel in self.config:
            hidden_dim = _apply_width(exp_size)
            out_channels = _apply_width(out_c)
            layers.append(GhostBottleneck(self.in_channels, hidden_dim, out_channels, dw_kernel_size=dw_kernel, stride=stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.bottlenecks(x)
        x = self.conv_head(x)
        x = self.bn_head(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.classifier_conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# ==========================================
# 2. VGG-16 for CIFAR
# ==========================================
class VGG16_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16_CIFAR, self).__init__()
        self.cfg = [
            64, 64, 'M',
            128, 128, 'M',
            256, 256, 256, 'M',
            512, 512, 512, 'M',
            512, 512, 512, 'M'
        ]
        self.features = self.make_layers(self.cfg)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

def replace_conv_with_ghost(model, ratio=2, dw_kernel_size=3):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size[0]
            stride = module.stride[0]
            ghost_layer = GhostModule(
                in_channels=in_channels,
                out_channels=out_channels,
                primary_kernel_size=kernel_size,
                ratio=ratio,
                dw_kernel_size=dw_kernel_size,
                stride=stride
            )
            setattr(model, name, ghost_layer)
        else:
            replace_conv_with_ghost(module, ratio, dw_kernel_size)

# ==========================================
# 3. Training Loop (Split: Train/Val/Test)
# ==========================================
def train_ghost_vgg():
    # Hyperparameters
    BATCH_SIZE = 128
    LR = 0.1
    EPOCHS = 200
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {DEVICE}")

    # Data Transforms
    print("Preparing CIFAR-10 Data...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 1. Load Full Training Data
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    # 2. Create Train/Validation Split (90/10)
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    train_subset, val_subset = random_split(full_trainset, [train_size, val_size])

    # 3. Create DataLoaders
    trainloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valloader = DataLoader(val_subset, batch_size=100, shuffle=False, num_workers=2)

    # Load separate Test Set (Held out until the very end)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # 4. Build Model
    print("\nBuilding Standard VGG-16...")
    net = VGG16_CIFAR()
    std_params = sum(p.numel() for p in net.parameters())
    print(f"Standard VGG Params: {std_params / 1e6:.2f}M")

    # 5. Swap Layers
    print("Swapping Conv2d with GhostModules...")
    replace_conv_with_ghost(net, ratio=2, dw_kernel_size=3)
    net = net.to(DEVICE)

    ghost_params = sum(p.numel() for p in net.parameters())
    print(f"Ghost-VGG Params: {ghost_params / 1e6:.2f}M")
    print(f"Reduction: {100 * (1 - ghost_params/std_params):.2f}%")

    # 6. Setup Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 7. Train Loop
    print(f"\nStarting Training on {len(train_subset)} samples, Validating on {len(val_subset)} samples...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Validation Step
        net.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in valloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/(batch_idx+1):.3f} | Val Acc: {val_acc:.2f}%")
        scheduler.step()

    print(f"Total training time: {time.time() - start_time:.1f}s")

    # 8. Final Evaluation on Test Set
    print("\nRunning Final Evaluation on Test Set...")
    net.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

    test_acc = 100. * test_correct / test_total
    print(f"Final Test Set Accuracy: {test_acc:.2f}%")

if __name__ == '__main__':
    train_ghost_vgg()


#----------------------------------------------------------------------------------------------
# Using device: cuda
# Preparing CIFAR-10 Data...
# 100%|██████████| 170M/170M [00:04<00:00, 40.3MB/s]

# Building Standard VGG-16...
# Standard VGG Params: 14.99M
# Swapping Conv2d with GhostModules...
# Ghost-VGG Params: 7.65M
# Reduction: 48.97%

# Starting Training on 45000 samples, Validating on 5000 samples...
# Epoch 1/200 | Train Loss: 2.131 | Val Acc: 19.44%
# Epoch 2/200 | Train Loss: 1.925 | Val Acc: 23.44%
# Epoch 3/200 | Train Loss: 1.763 | Val Acc: 34.06%
# Epoch 4/200 | Train Loss: 1.512 | Val Acc: 42.24%
# Epoch 5/200 | Train Loss: 1.276 | Val Acc: 51.58%
# Epoch 6/200 | Train Loss: 1.131 | Val Acc: 58.72%
# Epoch 7/200 | Train Loss: 0.970 | Val Acc: 64.24%
# Epoch 8/200 | Train Loss: 0.878 | Val Acc: 69.64%
# Epoch 9/200 | Train Loss: 0.820 | Val Acc: 75.66%
# Epoch 10/200 | Train Loss: 0.786 | Val Acc: 72.22%
# Epoch 11/200 | Train Loss: 0.730 | Val Acc: 71.34%
# Epoch 12/200 | Train Loss: 0.701 | Val Acc: 75.02%
# Epoch 13/200 | Train Loss: 0.688 | Val Acc: 71.12%
# Epoch 14/200 | Train Loss: 0.668 | Val Acc: 73.28%
# Epoch 15/200 | Train Loss: 0.652 | Val Acc: 73.76%
# Epoch 16/200 | Train Loss: 0.625 | Val Acc: 72.26%
# Epoch 17/200 | Train Loss: 0.618 | Val Acc: 77.18%
# Epoch 18/200 | Train Loss: 0.597 | Val Acc: 73.56%
# Epoch 19/200 | Train Loss: 0.589 | Val Acc: 73.34%
# Epoch 20/200 | Train Loss: 0.580 | Val Acc: 78.36%
# Epoch 21/200 | Train Loss: 0.577 | Val Acc: 78.42%
# Epoch 22/200 | Train Loss: 0.567 | Val Acc: 65.04%
# Epoch 23/200 | Train Loss: 0.561 | Val Acc: 72.54%
# Epoch 24/200 | Train Loss: 0.543 | Val Acc: 76.00%
# Epoch 25/200 | Train Loss: 0.549 | Val Acc: 74.44%
# Epoch 26/200 | Train Loss: 0.541 | Val Acc: 77.64%
# Epoch 27/200 | Train Loss: 0.538 | Val Acc: 77.46%
# Epoch 28/200 | Train Loss: 0.530 | Val Acc: 78.38%
# Epoch 29/200 | Train Loss: 0.534 | Val Acc: 75.62%
# Epoch 30/200 | Train Loss: 0.514 | Val Acc: 75.88%
# Epoch 31/200 | Train Loss: 0.523 | Val Acc: 81.68%
# Epoch 32/200 | Train Loss: 0.511 | Val Acc: 76.08%
# Epoch 33/200 | Train Loss: 0.510 | Val Acc: 71.88%
# Epoch 34/200 | Train Loss: 0.499 | Val Acc: 73.96%
# Epoch 35/200 | Train Loss: 0.507 | Val Acc: 76.94%
# Epoch 36/200 | Train Loss: 0.499 | Val Acc: 76.82%
# Epoch 37/200 | Train Loss: 0.502 | Val Acc: 78.88%
# Epoch 38/200 | Train Loss: 0.485 | Val Acc: 80.44%
# Epoch 39/200 | Train Loss: 0.488 | Val Acc: 79.40%
# Epoch 40/200 | Train Loss: 0.479 | Val Acc: 81.56%
# Epoch 41/200 | Train Loss: 0.476 | Val Acc: 75.04%
# Epoch 42/200 | Train Loss: 0.476 | Val Acc: 82.88%
# Epoch 43/200 | Train Loss: 0.472 | Val Acc: 76.64%
# Epoch 44/200 | Train Loss: 0.468 | Val Acc: 80.24%
# Epoch 45/200 | Train Loss: 0.462 | Val Acc: 75.40%
# Epoch 46/200 | Train Loss: 0.470 | Val Acc: 82.04%
# Epoch 47/200 | Train Loss: 0.463 | Val Acc: 79.22%
# Epoch 48/200 | Train Loss: 0.455 | Val Acc: 79.76%
# Epoch 49/200 | Train Loss: 0.449 | Val Acc: 78.70%
# Epoch 50/200 | Train Loss: 0.452 | Val Acc: 76.04%
# Epoch 51/200 | Train Loss: 0.454 | Val Acc: 84.08%
# Epoch 52/200 | Train Loss: 0.446 | Val Acc: 79.52%
# Epoch 53/200 | Train Loss: 0.446 | Val Acc: 82.68%
# Epoch 54/200 | Train Loss: 0.444 | Val Acc: 83.46%
# Epoch 55/200 | Train Loss: 0.443 | Val Acc: 72.30%
# Epoch 56/200 | Train Loss: 0.433 | Val Acc: 83.02%
# Epoch 57/200 | Train Loss: 0.436 | Val Acc: 76.78%
# Epoch 58/200 | Train Loss: 0.426 | Val Acc: 68.82%
# Epoch 59/200 | Train Loss: 0.431 | Val Acc: 78.26%
# Epoch 60/200 | Train Loss: 0.419 | Val Acc: 74.60%
# Epoch 61/200 | Train Loss: 0.420 | Val Acc: 82.36%
# Epoch 62/200 | Train Loss: 0.419 | Val Acc: 81.72%
# Epoch 63/200 | Train Loss: 0.413 | Val Acc: 79.50%
# Epoch 64/200 | Train Loss: 0.408 | Val Acc: 82.56%
# Epoch 65/200 | Train Loss: 0.423 | Val Acc: 81.76%
# Epoch 66/200 | Train Loss: 0.405 | Val Acc: 79.24%
# Epoch 67/200 | Train Loss: 0.402 | Val Acc: 75.38%
# Epoch 68/200 | Train Loss: 0.398 | Val Acc: 80.58%
# Epoch 69/200 | Train Loss: 0.401 | Val Acc: 83.72%
# Epoch 70/200 | Train Loss: 0.395 | Val Acc: 85.14%
# Epoch 71/200 | Train Loss: 0.386 | Val Acc: 83.80%
# Epoch 72/200 | Train Loss: 0.386 | Val Acc: 82.88%
# Epoch 73/200 | Train Loss: 0.396 | Val Acc: 80.34%
# Epoch 74/200 | Train Loss: 0.383 | Val Acc: 79.92%
# Epoch 75/200 | Train Loss: 0.382 | Val Acc: 82.82%
# Epoch 76/200 | Train Loss: 0.377 | Val Acc: 83.88%
# Epoch 77/200 | Train Loss: 0.379 | Val Acc: 81.06%
# Epoch 78/200 | Train Loss: 0.378 | Val Acc: 83.20%
# Epoch 79/200 | Train Loss: 0.368 | Val Acc: 81.20%
# Epoch 80/200 | Train Loss: 0.369 | Val Acc: 81.80%
# Epoch 81/200 | Train Loss: 0.366 | Val Acc: 83.18%
# Epoch 82/200 | Train Loss: 0.358 | Val Acc: 85.06%
# Epoch 83/200 | Train Loss: 0.358 | Val Acc: 80.54%
# Epoch 84/200 | Train Loss: 0.354 | Val Acc: 83.20%
# Epoch 85/200 | Train Loss: 0.346 | Val Acc: 84.58%
# Epoch 86/200 | Train Loss: 0.348 | Val Acc: 83.58%
# Epoch 87/200 | Train Loss: 0.340 | Val Acc: 83.98%
# Epoch 88/200 | Train Loss: 0.337 | Val Acc: 84.40%
# Epoch 89/200 | Train Loss: 0.333 | Val Acc: 83.76%
# Epoch 90/200 | Train Loss: 0.332 | Val Acc: 82.54%
# Epoch 91/200 | Train Loss: 0.337 | Val Acc: 85.28%
# Epoch 92/200 | Train Loss: 0.323 | Val Acc: 83.24%
# Epoch 93/200 | Train Loss: 0.323 | Val Acc: 81.78%
# Epoch 94/200 | Train Loss: 0.315 | Val Acc: 83.94%
# Epoch 95/200 | Train Loss: 0.327 | Val Acc: 83.46%
# Epoch 96/200 | Train Loss: 0.316 | Val Acc: 85.10%
# Epoch 97/200 | Train Loss: 0.305 | Val Acc: 86.66%
# Epoch 98/200 | Train Loss: 0.308 | Val Acc: 78.92%
# Epoch 99/200 | Train Loss: 0.303 | Val Acc: 84.40%
# Epoch 100/200 | Train Loss: 0.308 | Val Acc: 83.50%
# Epoch 101/200 | Train Loss: 0.298 | Val Acc: 82.60%
# Epoch 102/200 | Train Loss: 0.289 | Val Acc: 86.48%
# Epoch 103/200 | Train Loss: 0.294 | Val Acc: 85.20%
# Epoch 104/200 | Train Loss: 0.280 | Val Acc: 85.72%
# Epoch 105/200 | Train Loss: 0.277 | Val Acc: 83.24%
# Epoch 106/200 | Train Loss: 0.277 | Val Acc: 85.14%
# Epoch 107/200 | Train Loss: 0.275 | Val Acc: 85.08%
# Epoch 108/200 | Train Loss: 0.275 | Val Acc: 85.22%
# Epoch 109/200 | Train Loss: 0.270 | Val Acc: 83.62%
# Epoch 110/200 | Train Loss: 0.260 | Val Acc: 86.46%
# Epoch 111/200 | Train Loss: 0.261 | Val Acc: 86.72%
# Epoch 112/200 | Train Loss: 0.261 | Val Acc: 84.12%
# Epoch 113/200 | Train Loss: 0.254 | Val Acc: 85.28%
# Epoch 114/200 | Train Loss: 0.252 | Val Acc: 85.80%
# Epoch 115/200 | Train Loss: 0.242 | Val Acc: 84.18%
# Epoch 116/200 | Train Loss: 0.239 | Val Acc: 85.10%
# Epoch 117/200 | Train Loss: 0.238 | Val Acc: 87.42%
# Epoch 118/200 | Train Loss: 0.229 | Val Acc: 85.94%
# Epoch 119/200 | Train Loss: 0.228 | Val Acc: 88.86%
# Epoch 120/200 | Train Loss: 0.224 | Val Acc: 87.08%
# Epoch 121/200 | Train Loss: 0.222 | Val Acc: 88.08%
# Epoch 122/200 | Train Loss: 0.221 | Val Acc: 84.96%
# Epoch 123/200 | Train Loss: 0.213 | Val Acc: 85.38%
# Epoch 124/200 | Train Loss: 0.217 | Val Acc: 84.22%
# Epoch 125/200 | Train Loss: 0.207 | Val Acc: 88.28%
# Epoch 126/200 | Train Loss: 0.203 | Val Acc: 86.86%
# Epoch 127/200 | Train Loss: 0.197 | Val Acc: 87.98%
# Epoch 128/200 | Train Loss: 0.194 | Val Acc: 88.26%
# Epoch 129/200 | Train Loss: 0.190 | Val Acc: 87.24%
# Epoch 130/200 | Train Loss: 0.187 | Val Acc: 87.54%
# Epoch 131/200 | Train Loss: 0.181 | Val Acc: 87.22%
# Epoch 132/200 | Train Loss: 0.173 | Val Acc: 87.90%
# Epoch 133/200 | Train Loss: 0.173 | Val Acc: 86.30%
# Epoch 134/200 | Train Loss: 0.169 | Val Acc: 88.66%
# Epoch 135/200 | Train Loss: 0.169 | Val Acc: 88.22%
# Epoch 136/200 | Train Loss: 0.162 | Val Acc: 87.06%
# Epoch 137/200 | Train Loss: 0.153 | Val Acc: 88.74%
# Epoch 138/200 | Train Loss: 0.151 | Val Acc: 88.12%
# Epoch 139/200 | Train Loss: 0.152 | Val Acc: 90.08%
# Epoch 140/200 | Train Loss: 0.141 | Val Acc: 87.58%
# Epoch 141/200 | Train Loss: 0.141 | Val Acc: 89.38%
# Epoch 142/200 | Train Loss: 0.135 | Val Acc: 88.64%
# Epoch 143/200 | Train Loss: 0.127 | Val Acc: 88.46%
# Epoch 144/200 | Train Loss: 0.132 | Val Acc: 88.30%
# Epoch 145/200 | Train Loss: 0.123 | Val Acc: 88.58%
# Epoch 146/200 | Train Loss: 0.117 | Val Acc: 88.72%
# Epoch 147/200 | Train Loss: 0.113 | Val Acc: 88.76%
# Epoch 148/200 | Train Loss: 0.110 | Val Acc: 89.50%
# Epoch 149/200 | Train Loss: 0.108 | Val Acc: 89.84%
# Epoch 150/200 | Train Loss: 0.102 | Val Acc: 89.88%
# Epoch 151/200 | Train Loss: 0.099 | Val Acc: 87.88%
# Epoch 152/200 | Train Loss: 0.094 | Val Acc: 89.64%
# Epoch 153/200 | Train Loss: 0.090 | Val Acc: 89.56%
# Epoch 154/200 | Train Loss: 0.090 | Val Acc: 89.28%
# Epoch 155/200 | Train Loss: 0.081 | Val Acc: 89.94%
# Epoch 156/200 | Train Loss: 0.073 | Val Acc: 90.80%
# Epoch 157/200 | Train Loss: 0.071 | Val Acc: 90.50%
# Epoch 158/200 | Train Loss: 0.071 | Val Acc: 90.64%
# Epoch 159/200 | Train Loss: 0.068 | Val Acc: 90.42%
# Epoch 160/200 | Train Loss: 0.062 | Val Acc: 89.68%
# Epoch 161/200 | Train Loss: 0.058 | Val Acc: 90.36%
# Epoch 162/200 | Train Loss: 0.052 | Val Acc: 90.42%
# Epoch 163/200 | Train Loss: 0.051 | Val Acc: 90.58%
# Epoch 164/200 | Train Loss: 0.048 | Val Acc: 91.20%
# Epoch 165/200 | Train Loss: 0.043 | Val Acc: 91.08%
# Epoch 166/200 | Train Loss: 0.038 | Val Acc: 90.12%
# Epoch 167/200 | Train Loss: 0.038 | Val Acc: 91.60%
# Epoch 168/200 | Train Loss: 0.038 | Val Acc: 90.94%
# Epoch 169/200 | Train Loss: 0.033 | Val Acc: 91.24%
# Epoch 170/200 | Train Loss: 0.028 | Val Acc: 91.24%
# Epoch 171/200 | Train Loss: 0.025 | Val Acc: 91.64%
# Epoch 172/200 | Train Loss: 0.023 | Val Acc: 91.32%
# Epoch 173/200 | Train Loss: 0.021 | Val Acc: 91.90%
# Epoch 174/200 | Train Loss: 0.017 | Val Acc: 91.40%
# Epoch 175/200 | Train Loss: 0.016 | Val Acc: 91.38%
# Epoch 176/200 | Train Loss: 0.016 | Val Acc: 92.12%
# Epoch 177/200 | Train Loss: 0.016 | Val Acc: 91.38%
# Epoch 178/200 | Train Loss: 0.013 | Val Acc: 91.90%
# Epoch 179/200 | Train Loss: 0.011 | Val Acc: 92.20%
# Epoch 180/200 | Train Loss: 0.009 | Val Acc: 92.24%
# Epoch 181/200 | Train Loss: 0.008 | Val Acc: 91.82%
# Epoch 182/200 | Train Loss: 0.008 | Val Acc: 92.26%
# Epoch 183/200 | Train Loss: 0.006 | Val Acc: 92.08%
# Epoch 184/200 | Train Loss: 0.006 | Val Acc: 92.38%
# Epoch 185/200 | Train Loss: 0.006 | Val Acc: 92.38%
# Epoch 186/200 | Train Loss: 0.005 | Val Acc: 92.28%
# Epoch 187/200 | Train Loss: 0.005 | Val Acc: 92.30%
# Epoch 188/200 | Train Loss: 0.004 | Val Acc: 92.34%
# Epoch 189/200 | Train Loss: 0.005 | Val Acc: 92.40%
# Epoch 190/200 | Train Loss: 0.003 | Val Acc: 92.48%
# Epoch 191/200 | Train Loss: 0.003 | Val Acc: 92.64%
# Epoch 192/200 | Train Loss: 0.003 | Val Acc: 92.70%
# Epoch 193/200 | Train Loss: 0.004 | Val Acc: 92.42%
# Epoch 194/200 | Train Loss: 0.003 | Val Acc: 92.44%
# Epoch 195/200 | Train Loss: 0.003 | Val Acc: 92.90%
# Epoch 196/200 | Train Loss: 0.003 | Val Acc: 92.76%
# Epoch 197/200 | Train Loss: 0.003 | Val Acc: 92.36%
# Epoch 198/200 | Train Loss: 0.003 | Val Acc: 92.54%
# Epoch 199/200 | Train Loss: 0.002 | Val Acc: 92.34%
# Epoch 200/200 | Train Loss: 0.003 | Val Acc: 92.48%
# Total training time: 4382.5s

# Running Final Evaluation on Test Set...
# Final Test Set Accuracy: 92.89%

#--------------------------------------------------------------------------------------------

# Initializing FLOPs measurement...for train_val_test split based on dummy input
# Measuring Standard VGG-16...
# Measuring Ghost-VGG-16...

# ==================================================
# Model                | FLOPs           | Params         
# --------------------------------------------------
# Standard VGG-16      | 314.570M        | 14.991M        
# Ghost-VGG-16         | 159.216M        | 7.650M         
# ==================================================

# Reduction Results:
# FLOPs Reduction:     49.39% (approx 1.98x speedup)
# Parameter Reduction: 48.97% (approx 1.96x smaller)

