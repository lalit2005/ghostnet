import json
import matplotlib.pyplot as plt
import pandas as pd

def plot_training():
    with open("experiment_results.json", "r") as f:
        data = json.load(f)

    plt.figure(figsize=(10, 6))
    
    styles = {
        "FasterRCNN_Run1": {"color": "blue", "label": "Ghost Faster R-CNN (Run 1)", "marker": "o"},
        "FasterRCNN_Run2": {"color": "cyan", "label": "Ghost Faster R-CNN (Run 2)", "marker": "x"},
        "RetinaNet_Run1":  {"color": "red",  "label": "Ghost RetinaNet (Run 1)", "marker": "s"}
    }

    for run_name, points in data.items():
        # Sort by epoch
        points.sort(key=lambda x: x['epoch'])
        epochs = [p['epoch'] for p in points]
        maps = [p['mAP'] * 100 for p in points] # Convert to %
        
        style = styles.get(run_name, {})
        plt.plot(epochs, maps, 
                 label=style.get("label", run_name), 
                 color=style.get("color"), 
                 marker=style.get("marker"))

    plt.title("GhostNet Detection Performance: Faster R-CNN vs RetinaNet", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("mAP (%) (COCO val2017)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ghostnet_detection_results.png", dpi=300)
    print("Plot saved to ghostnet_detection_results.png")

if __name__ == "__main__":
    plot_training()
