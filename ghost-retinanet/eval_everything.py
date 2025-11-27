import torch
import os
import json
import glob
import sys

# --- IMPORTS BASED ON YOUR FILE TREE ---
# Make sure these filenames match exactly what is in your folder
try:
    from ghostnet_faster_rcnn import get_ghost_detection_model, CocoDetectionWrapper, collate_fn
    from retinanet import get_ghost_retinanet 
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure 'ghostnet_faster_rcnn.py' and 'retinanet.py' are in the same folder as this script.")
    sys.exit(1)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torchvision.transforms as transforms

# --- CONFIGURATION ---
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# verify data paths
val_dir = 'data/coco/val2017'
val_ann = 'data/coco/annotations/instances_val2017.json'

if not os.path.exists(val_ann):
    print(f"ERROR: Annotations not found at {val_ann}")
    sys.exit(1)

# --- UPDATED PATHS TO MATCH YOUR TREE ---
run_configs = {
    "FasterRCNN_Run1": {
        # Looking inside checkpoints/faster-rcnn-run-1/
        "files": sorted(glob.glob("checkpoints/faster-rcnn-run-1/ghost_fasterrcnn_epoch_*.pth")),
        "type": "rcnn"
    },
    "FasterRCNN_Run2": {
        # Looking inside checkpoints/faster-rcnn-run-2/
        "files": sorted(glob.glob("checkpoints/faster-rcnn-run-2/ghost_fasterrcnn_epoch_*.pth")),
        "type": "rcnn"
    },
    "RetinaNet_Run1": {
        # Looking inside checkpoints/retinanet-run-1/
        "files": sorted(glob.glob("checkpoints/retinanet-run-1/ghost_retinanet_epoch_*.pth")),
        "type": "retina"
    }
}

def evaluate_checkpoint(model, dataloader, coco_gt):
    model.eval()
    results = []
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            for target, output in zip(targets, outputs):
                image_id = target["image_id"].item()
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                
                for box, score, label in zip(boxes, scores, labels):
                    x, y, x2, y2 = box
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [float(x), float(y), float(x2-x), float(y2-y)],
                        "score": float(score)
                    })
            
            # Print progress every 100 batches so you know it's working
            if i % 100 == 0:
                print(f"   Batch {i} processed...", end="\r")
    
    if not results: return 0.0
    
    # Suppress COCO print output to keep logs clean
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    
    # Return mAP (IoU=0.50:0.95) which is stats[0]
    return coco_eval.stats[0] 

def main():
    print(f"Using device: {device}")
    
    # Setup Data
    print("Loading Validation Data...")
    dataset = CocoDetectionWrapper(root=val_dir, annFile=val_ann, transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn)
    coco_gt = COCO(val_ann)
    
    final_stats = {}

    for run_name, config in run_configs.items():
        print(f"\n==========================================")
        print(f" Processing Group: {run_name}")
        print(f" Found {len(config['files'])} checkpoint files.")
        print(f"==========================================")
        
        if len(config['files']) == 0:
            print("!! Warning: No files found. Check directory paths.")
            continue

        stats = []
        
        # Initialize Model Structure
        if config["type"] == "rcnn":
            model = get_ghost_detection_model(num_classes=91)
        else:
            model = get_ghost_retinanet(num_classes=91)
        
        model.to(device)
        
        for file_path in config["files"]:
            print(f"Evaluating {os.path.basename(file_path)}...", end=" ")
            
            try:
                state_dict = torch.load(file_path, map_location=device)
                model.load_state_dict(state_dict)
                
                # Run Eval
                mAP = evaluate_checkpoint(model, dataloader, coco_gt)
                print(f"-> mAP: {mAP:.4f}")
                
                # Extract epoch number safely
                # Splitting by '_' and taking the last part before .pth
                # e.g. "ghost_fasterrcnn_epoch_10.pth" -> "10"
                base = os.path.basename(file_path)
                epoch_str = base.replace(".pth", "").split("_")[-1]
                epoch_num = int(epoch_str)
                
                stats.append({"epoch": epoch_num, "mAP": mAP, "file": file_path})
                
            except Exception as e:
                print(f"\n!! Failed to evaluate {file_path}: {e}")

        final_stats[run_name] = stats

        # Save PARTIAL results after every run group (so you don't lose data if it crashes)
        with open("experiment_results.json", "w") as f:
            json.dump(final_stats, f, indent=4)

    print("\nAll Done! Final results saved to experiment_results.json")

if __name__ == "__main__":
    main()
