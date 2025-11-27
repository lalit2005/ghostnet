import torch
import torchvision
from ghostnet_faster_rcnn import get_ghost_detection_model, CocoDetectionWrapper, collate_fn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import os

def evaluate_map(checkpoint_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Loading checkpoint: {checkpoint_path}...")
    
    # coco has 91 classes
    model = get_ghost_detection_model(num_classes=91)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    val_dir = 'data/coco/val2017'
    val_ann = 'data/coco/annotations/instances_val2017.json'
    
    if not os.path.exists(val_ann):
        print(f"Error: Annotation file not found at {val_ann}")
        return

    dataset = CocoDetectionWrapper(root=val_dir, annFile=val_ann, transform=torchvision.transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    coco_gt = COCO(val_ann)
    results = []

    print("Running inference on validation set (5000 images)...")
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            images = list(img.to(device) for img in images)
            
            outputs = model(images)
            
            # converting to coco format
            for target, output in zip(targets, outputs):
                image_id = target["image_id"].item()
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                
                for box, score, label in zip(boxes, scores, labels):
                    x, y, x2, y2 = box
                    # convert x,y,x,y -> x,y,w,h,
                    w = x2 - x
                    h = y2 - y
                    
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "score": float(score)
                    })
            
            if i % 50 == 0:
                print(f"Processed batch {i}/{len(dataloader)}")

    print("Calculating mAP...")
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    evaluate_map("checkpoints/ghost_fasterrcnn_epoch_12.pth")
