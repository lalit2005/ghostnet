import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork
import os
import math
import time
from PIL import Image

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, primary_kernel_size=1, ratio=2, dw_kernel_size=3, stride=1):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Conv2d(in_channels, init_channels, primary_kernel_size, stride, (primary_kernel_size-1)//2, bias=False)
        self.cheap_operation = nn.Conv2d(init_channels, new_channels, dw_kernel_size, 1, (dw_kernel_size-1)//2, groups=init_channels, bias=False)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]

class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dw_kernel_size, stride):
        super(GhostBottleneck, self).__init__()
        self.ghost1 = GhostModule(in_channels, hidden_dim, 1, 2, dw_kernel_size)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        
        if stride == 2:
            self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, dw_kernel_size, stride, (dw_kernel_size-1)//2, groups=hidden_dim, bias=False)
            self.bn_dw = nn.BatchNorm2d(hidden_dim)
        else:
            self.dwconv = None

        self.ghost2 = GhostModule(hidden_dim, out_channels, 1, 2, dw_kernel_size)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, dw_kernel_size, stride, (dw_kernel_size-1)//2, groups=in_channels, bias=False) if stride==2 else nn.Identity(),
                nn.BatchNorm2d(in_channels) if stride==2 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.ghost1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.dwconv:
            x = self.dwconv(x)
            x = self.bn_dw(x)
        x = self.ghost2(x)
        x = self.bn2(x)
        return x + identity

class GhostNetBackbone(nn.Module):
    def __init__(self, width_mult=1.0):
        super(GhostNetBackbone, self).__init__()
        self.in_channels = 16
        def make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v: new_v += divisor
            return new_v
        
        init_conv_out = make_divisible(16 * width_mult)
        self.conv1 = nn.Conv2d(3, init_conv_out, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(init_conv_out)
        self.relu = nn.ReLU(inplace=True)
        self.in_channels = init_conv_out

        # config from table 1
        self.config = [
            [3, 16, 16, 0, 1], [3, 48, 24, 0, 2], [3, 72, 24, 0, 1], 
            [5, 72, 40, 1, 2], [5, 120, 40, 1, 1], [3, 240, 80, 0, 2], 
            [3, 200, 80, 0, 1], [3, 184, 80, 0, 1], [3, 184, 80, 0, 1], 
            [3, 480, 112, 1, 1], [3, 672, 112, 1, 1], [5, 672, 160, 1, 2], 
            [5, 960, 160, 0, 1], [5, 960, 160, 1, 1], [5, 960, 160, 0, 1], 
            [5, 960, 160, 1, 1] 
        ]
        self.layers = nn.ModuleList()
        for k, exp, c, se, s in self.config:
            out_c = make_divisible(c * width_mult)
            hidden_c = make_divisible(exp * width_mult)
            self.layers.append(GhostBottleneck(self.in_channels, hidden_c, out_c, k, s))
            self.in_channels = out_c

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features = {}
        # obtain features at strides 8, 16, 32 for fpn
        indices = {3: "0", 5: "1", 11: "2"} 
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in indices:
                features[indices[i]] = x
        return features

# ===================================
# ===================================
# ===================================
# retinanet + fpn
# ===================================

def get_ghost_retinanet(num_classes):
    backbone = GhostNetBackbone()
    
    # ghostnet returns features at stride 8, 16, 32. 
    backbone_with_fpn = torchvision.ops.FeaturePyramidNetwork(
        in_channels_list=[40, 80, 160], 
        out_channels=256
    )
    
    class BackboneWithFPN(nn.Module):
        def __init__(self, bb, fpn):
            super().__init__()
            self.body = bb
            self.fpn = fpn
            self.out_channels = 256
        def forward(self, x):
            x = self.body(x)
            x = self.fpn(x)
            return x

    full_backbone = BackboneWithFPN(backbone, backbone_with_fpn)
    
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,)), 
        aspect_ratios=((0.5, 1.0, 2.0),) * 3
    )
    
    model = RetinaNet(
        full_backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator
    )
    return model

class CocoDetectionWrapper(torchvision.datasets.CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super(CocoDetectionWrapper, self).__init__(root, annFile)
        self.transform = transform

    def __getitem__(self, idx):
        img, target = super(CocoDetectionWrapper, self).__getitem__(idx)
        image_id = self.ids[idx]
        
        boxes = []
        labels = []
        
        for obj in target:
            if 'bbox' in obj:
                x, y, w, h = obj['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(obj['category_id'])

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            labels = labels[keep]
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0), dtype=torch.int64)

        target_dict = {}
        target_dict["boxes"] = boxes
        target_dict["labels"] = labels
        target_dict["image_id"] = torch.tensor([image_id])

        if self.transform:
            img = self.transform(img)

        return img, target_dict

def collate_fn(batch):
    return tuple(zip(*batch))


# ===================================
# ===================================
# training loop
# ===================================
if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Paths (Assumed from previous setup)
    train_dir = 'data/coco/train2017'
    train_ann = 'data/coco/annotations/instances_train2017.json'
    
    transform = transforms.Compose([transforms.ToTensor()])

    print("Loading COCO dataset...")
    full_dataset = CocoDetectionWrapper(root=train_dir, annFile=train_ann, transform=transform)
    
    # using small validation set from train2017 dataset
    total_size = len(full_dataset)
    indices = torch.randperm(total_size).tolist()
    
    train_size = total_size - 2000
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    BATCH_SIZE = 8 
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    print("building ghost retinanet...")
    model = get_ghost_retinanet(num_classes=91)
    
    # imaenet weights loading for the backbone
    weights_path = 'ghostnet_1x_imagenet.pth'
    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.backbone.body.load_state_dict(state_dict, strict=False)
            print(">> Loaded pre-trained GhostNet backbone weights.")
        except Exception as e:
            print(f">> Warning: Failed to load weights: {e}")
    else:
        print(">> No pre-trained backbone weights found. Training from scratch (performance will be lower).")
    # ---------------------------------

    model.to(device)

    # paper uses sgd for 12 epochs 
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 11], gamma=0.1)

    print("\n-----------------------------------------------------------")
    print(f"STARTING GHOST-RETINANET TRAINING (Batch Size: {BATCH_SIZE})")
    print("Press Ctrl+C at any time to stop and save.")
    print("-----------------------------------------------------------\n")

    num_epochs = 12
    model.train()

    try:
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            epoch_loss = 0
            
            for i, (images, targets) in enumerate(train_loader):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # RetinaNet returns a dict of losses: 'classification', 'bbox_regression'
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                if not math.isfinite(losses.item()):
                    print(f"Loss is {losses.item()}, stopping")
                    print(loss_dict)
                    sys.exit(1)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                epoch_loss += losses.item()

                if i % 100 == 0:
                    print(f"Iter {i}: Loss {losses.item():.4f}")
            
            scheduler.step()
            print(f"Epoch {epoch+1} Complete. Avg Loss: {epoch_loss/len(train_loader):.4f}")
            
            # Save checkpoint
            torch.save(model.state_dict(), f"ghost_retinanet_epoch_{epoch+1}.pth")

    except KeyboardInterrupt:
        print("\n\n[!] Interrupted. Saving state...")
        torch.save(model.state_dict(), "ghost_retinanet_interrupted.pth")
    
    print("Done.")
