## ghostnet

implementation of the paper [ghostnet paper](https://arxiv.org/abs/1911.11907)

### running locally

- start a virtual env with `python3 -m venv .`
- run `python ghostnet.py`
- to download ghostnet weights pretrained on imagenet, run `python download-imagenet-weights.py`

### experiments

### ghostnet-resnet-56

| no. | ratio(s) | kernel(d) | optimizer | lr_scheduler          | accuracy(ours) | accuracy(paper) | epochs | file                  |
| --- | -------- | --------- | --------- | --------------------- | -------------- | --------------- | ------ | --------------------- |
| 1   | 2        | 3         | SGD       | CosineAnnealing       | 92.20%         | 92.7%           | 200    | gr_cosine.pth         |
| 2   | 2        | 3         | SGD       | MultiStepLR(100, 500) | 92.55%         | 92.7%           | 200    | gr_multistep.pth      |
| 3   | 2        | 3         | SGD       | CosineAnnealing.      | 91.30%         | 92.7%           | 200    | ghost-resnet-trial.py |

- total feature maps = s \* number of intrinsic feature maps (produced by ordinary convolution filters)
- `images/ghost_visualization.png` (dog's image, cifar index=12) proves that ghost map preserves the exact same shape and pose as the intrinsic maps
- the ghost versions appear slightly shifted and smoothened. this confirms that the cheap operation (depthwise convolutions) learned useful transformations like blurring and edge enhancement to augment the feature space without the computationally heavy full convolution operation.
- computational reduction(measured using `thops`):

```
measuring standard resnet-56...
  flops: 127.93M
  params: 0.86M

measuring ghost-resnet-56...
  flops: 67.50M
  params: 0.44M

results:
  flops reduction: 1.90x (paper claims ~2x)
  param reduction: 1.95x (paper claims ~2x)
```

---

### ghost-vgg-16

experiment run on cifar-10 using a dynamic layer replacement strategy (swapping `nn.conv2d` for `ghostmodule`).

| no. | ratio(s) | kernel(d) | optimizer | lr_scheduler    | accuracy(ours) | accuracy(paper) | epochs | file                        |
| --- | -------- | --------- | --------- | --------------- | -------------- | --------------- | ------ | --------------------------- |
| 1   | 2        | 3         | SGD       | CosineAnnealing | 93.63%         | 93.7%           | 200    | ghost-vgg.py                |
| 2   | 2        | 3         | SGD       | CosineAnnealing | 92.83%         | 93.7%           | 200    | ghost-vgg-trial.py          |
| 3   | 2        | 3         | SGD       | CosineAnnealing | 92.89%         | 93.7%           | 200    | ghost_vgg_with_val_split.py |

- **convergence**: achieved 93.60% accuracy at epoch 200. best accuracy observed was ~93.63%.
- the third run used train-val-test dataset splits, though the paper & other 2 runs use train-test split
- **computational reduction**:

```
calculating flops...
standard vgg-16: 314.570M FLOPs | 14.991M Params
ghost-vgg-16:    159.216M FLOPs | 7.650M Params
----------------------------------------
flops reduction:     49.39% (approx 1.98x)
parameter reduction: 48.97% (approx 1.96x)
```

---

### faster-r-cnn with ghostnet backbone

| no. | ratio(s) | kernel(d)     | optimizer | lr_scheduler     | mAP            | mAP             | epochs | file                    |
| --- | -------- | ------------- | --------- | ---------------- | -------------- | --------------- | ------ | ----------------------- |
| 1   | 2        | 3,5(varied)   | sgd       | constant(0.005)  | 8.5%           | 26.9%           | 12     | ghostnet_faster_rcnn    |
| 2   | 2        | 3, 5 (varied) | sgd       | constant (0.005) | 9.4%           | 26.9%           | 9      | `faster-rcnn-run-2/...` |

- the models mentioned in the paper used sgd for 12 epochs from imagenet pretrained weights, but our model was trained from scratch. hence there's a huge difference in map between the paper's results and ours
- **note on run 2:** run 2 utilized a larger batch size `b=8` resulting in faster convergence and slightly higher accuracy (9.4%) in fewer epochs compared to run 1.
- the paper used 1.1 as the width multiplier, but we used 1.0 as the width multiplier to decrease the computation required
- but the decrease in computation between mobile net models is consistent with the paper. the model uses half the compute of what mobilenet uses. our model can match the results of mobilenet if trained for more epochs

```
ghostnet backbone (1.0x) flops: 149.31m
ghostnet backbone (1.0x) params: 1.06m

--- paper comparison (table 8) ---
mobilenetv2 backbone flops:  300m
mobilenetv3 backbone flops:  219m
ghostnet 1.1x backbone flops: 164m
```

---

### retina net with ghostnet backbone

To test the backbone's versatility, we integrated it into a One-Stage Detector (RetinaNet).

| No. | Ratio(s) | Kernel(d) | Optimizer | LR Scheduler | Accuracy (Ours) | Epochs | File            |
| --- | -------- | --------- | --------- | ------------ | --------------- | ------ | --------------- |
| 1   | 2        | 3,5       | SGD       | MultiStep    | 2.58% mAP\*     | 6      | ghost_retinanet |

- training was halted at epoch 6 due to time constraints. the learning curve (see plots) shows consistent convergence, verifying the backbone works for one-stage detectors.

---

## ablation study

we extended the paper's analysis by testing extreme ratios `s` on resnet-56 to find the limit of redundancy.

<img width="897" height="226" alt="image" src="https://github.com/user-attachments/assets/d8afc9ad-5aeb-4ebe-a0f7-987316c182c9" />

> **discovery:** we found that we can remove 90% of the standard convolution calculations `s=10` and replace them with cheap linear operations while losing less than 4.5% accuracy. this suggests massive redundancy in standard resnet features.
