import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.optim as optim
import time
import torchvision
import torchvision.transforms as transforms

class GhostModule(nn.Module):
  def __init__(self, in_channels, out_channels, primary_kernel_size=1, ratio = 2, dw_kernel_size=3, stride=1):
    super(GhostModule, self).__init__()
    self.ratio = ratio
    self.out_channels = out_channels

    self.init_channels = math.ceil(out_channels / ratio) # num of intrinsic channels
    self.new_channels = self.init_channels * (self.ratio - 1)

    self.primary_conv = nn.Conv2d(
       in_channels,
       self.init_channels,
       kernel_size = primary_kernel_size,
       stride = stride,
       padding = (primary_kernel_size - 1) // 2,
       bias = False
    )

    self.cheap_operations = nn.Conv2d(
      in_channels = self.init_channels,
      out_channels = self.new_channels,
      kernel_size = dw_kernel_size,
      stride = 1,
      padding = (dw_kernel_size - 1) // 2,
      groups = self.init_channels,
      bias = False
    )

  def forward(self, x):
    intrinsic_features = self.primary_conv(x)
    cheap_features = self.cheap_operations(intrinsic_features)
    out = torch.cat([intrinsic_features, cheap_features], dim = 1)
    return out[:, :self.out_channels, :, :]

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        # if input and output dimensions don't match,
        # use 1x1 conv to project the shortcut
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out;


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes = 10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self.make_layer(block, 16, num_blocks[0], stride = 1)
        self.layer2 = self.make_layer(block, 32, num_blocks[1], stride = 2)
        self.layer3 = self.make_layer(block, 64, num_blocks[2], stride = 2)

        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet56():
    # 1 conv1 + 3 stages * 9 blocks/stage * 2 convs/block + 1 (linear) = 56 layers
    return ResNet(BasicBlock, [9, 9, 9])

def convert_to_ghost_resnet(model, ratio=0, dw_kernel_size=3):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size[0]
            stride = module.stride[0]

            new_ghost_module = GhostModule(
                in_channels = in_channels,
                out_channels = out_channels,
                primary_kernel_size = kernel_size,
                ratio = ratio,
                dw_kernel_size = dw_kernel_size,
                stride = stride
            )

            setattr(model, name, new_ghost_module)

        elif len(list(module.children())) > 0:
            convert_to_ghost_resnet(module, ratio, dw_kernel_size)


print("======================================")
print("ResNet56")
print("======================================")
model = ResNet56()
# # print(model)
#
convert_to_ghost_resnet(model, ratio = 20, dw_kernel_size = 3)
#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
#
optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 1e-4)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 200)
criterion = nn.CrossEntropyLoss()
#
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

batch_size = 128

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

print("Data loaders created.")

def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_id, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_id, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print(f'Epoch {epoch} Test Acc: {acc:.3f}%')
    return acc



print(f"Starting training on {device} for 200 epochs...")

best_acc = 0.0

for epoch in range(200):
    train(epoch)
    acc = test(epoch)
    scheduler.step()

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'gr_cosine.pth')

print(f"\nFinal Best Accuracy: {best_acc:.3f}%")
print(f"Paper Target: 92.7%")



# best_acc = 0.0
#
# print("======================================")
# print("STARTING TRAINING")
# print("======================================")
#
# besct_acc = 0.0
#
# for epoch in range(200):
#     train(epoch)
#     acc = test(epoch)
#     scheduler.step()
#     if acc > best_acc:
#         best_acc = acc
#         torch.save(model.state_dict(), 'gr_multistep.pth')
#         print(f"saved highest accuracy yet - {best_acc}")
#
#
# print("======================================")
# print(f"FINAL BEST ACCURACY: {best_acc}")
# print("======================================")

# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import numpy
#
# def visualize_ghost_module(model, image_index=0):
#     model.eval()
#     image, label = testset[image_index]
#     input_tensor = image.unsqueeze(0).to(device)
#     target_module = model.layer1[0].conv1 # this is the first ghost module of the network
#     feature_maps = {}
#
#     def hook_fn(module, input, output):
#         x = input[0]
#         x1 = module.primary_conv(x)
#         x2 = module.cheap_operations(x1)
#         feature_maps['intrinsic'] = x1.detach().cpu()
#         feature_maps['ghost'] = x2.detach().cpu()
#
#     handle = target_module.register_forward_hook(hook_fn)
#     with torch.no_grad():
#         model(input_tensor)
#     handle.remove()
#     return feature_maps, image
#
# def plot_and_save_ghosts(maps, filename="images/ghost_visualization.png"):
#     intrinsic = maps['intrinsic'][0]
#     ghosts = maps['ghost'][0]
#     num_pairs = min(len(intrinsic), len(ghosts))
#     fig, axes = plt.subplots(num_pairs, 2, figsize = (6, 2 * num_pairs))
#     fig.suptitle(f"Intrinsic vs Ghost Features (Layer 1)", fontsize=16)
#
#     for i in range(num_pairs):
#         # plotting intrinsic
#         axes[i, 0].imshow(intrinsic[i], cmap='viridis')
#         axes[i, 0].axis('off')
#         if i == 0: axes[i, 0].set_title("Intrinsic (Primary)")
#
#         # plotting ghost
#         axes[i, 1].imshow(ghosts[i], cmap='viridis')
#         axes[i, 1].axis('off')
#         if i == 0: axes[i, 1].set_title("Ghost (Cheap Op)")
#
#     plt.tight_layout()
#     plt.savefig(filename)
#     print(f"\n[Success] Visualization saved to: {filename}")
#     plt.close(fig)
#
# if __name__ == "__main__":
#     model.load_state_dict(torch.load('./gr_multistep.pth'))
#     print("extracting features...")
#     maps, raw_img = visualize_ghost_module(model, image_index=20)
#     plot_and_save_ghosts(maps, "images/ghost_resnet_image_20")



# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================

# CREATING STANDARD RESNET-56...
# CREATING GHOST-RESNET-56 (S=2)...
#
# MEASURING STANDARD RESNET-56...
#   FLOPS: 127.93 M
#   PARAMS: 0.86 M
#
# MEASURING GHOST-RESNET-56...
#   FLOPS: 67.50 M
#   PARAMS: 0.44 M
#
# RESULTS:
#   FLOPS REDUCTION: 1.90X (PAPER CLAIMS ~2X)
#   PARAM REDUCTION: 1.95X (PAPER CLAIMS ~2X)

# import torch
# from thop import profile
# from copy import deepcopy
#
# def compare_efficiency():
#     # mimicing dummy cifar-10 input
#     input_tensor = torch.randn(1, 3, 32, 32)
#
#     print("Creating and measuring Standard ResNet-56...")
#     standard_model = ResNet56()
#     flops_std, params_std = profile(standard_model, inputs=(input_tensor, ), verbose=False)
#
#     print(f"  [Standard] FLOPs: {flops_std / 1e6:.2f} M")
#     print(f"  [Standard] Params: {params_std / 1e6:.2f} M")
#     print("-" * 50)
#
#     ratios_to_test = [2, 3, 5, 10]
#
#     for s in ratios_to_test:
#         print(f"\nCreating Ghost-ResNet-56 (ratio s={s})...")
#
#         ghost_model = deepcopy(standard_model)
#         convert_to_ghost_resnet(ghost_model, ratio=s, dw_kernel_size=3)
#
#         flops_ghost, params_ghost = profile(ghost_model, inputs=(input_tensor, ), verbose=False)
#
#         flops_reduction = flops_std / flops_ghost
#         param_reduction = params_std / params_ghost
#
#         print(f"  [s={s}] FLOPs: {flops_ghost / 1e6:.2f} M")
#         print(f"  [s={s}] Params: {params_ghost / 1e6:.2f} M")
#         print(f"  >>> FLOPs Reduction: {flops_reduction:.2f}x")
#         print(f"  >>> Param Reduction: {param_reduction:.2f}x")
#
# if __name__ == "__main__":
#     compare_efficiency()
