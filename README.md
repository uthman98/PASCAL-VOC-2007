# Faster R-CNN Object Detection on PASCAL VOC 2007

An end-to-end object detection pipeline in PyTorch. A Faster R-CNN model (ResNet-50 + FPN backbone) pre-trained on COCO is fine-tuned on the 20 object classes of the PASCAL VOC 2007 dataset. The trained model is then used to detect objects in validation images and in your own images.

Everything is in a single notebook: [PASCAL_VOC_2007.ipynb](PASCAL_VOC_2007.ipynb).

## What the notebook does

1. **Data acquisition**: downloads the VOC 2007 `trainval` split (5,011 images) with `torchvision.datasets.VOCDetection` and plots ground-truth boxes so you can check the annotations.
2. **Preprocessing**: a custom `Dataset` turns the VOC XML annotations into the target format torchvision detection models expect (`boxes`, `labels`, `image_id`, `area`, `iscrowd`). Bounding-box-aware augmentations from `torchvision.transforms.v2` are applied: random horizontal flip, color jitter and removal of degenerate boxes. The data is split 80/20 into training (4,008 images, 12,635 boxes) and validation (1,003 images).
3. **Model**: `fasterrcnn_resnet50_fpn` with COCO weights. The box predictor head is replaced to output 21 classes (20 VOC classes + background).
4. **Training**:
   - SGD (lr 0.005, momentum 0.9, weight decay 5e-4)
   - `StepLR` scheduler (×0.1 every 5 epochs)
   - batch size 4 with 2-step gradient accumulation (effective batch size 8)
   - 15 epochs
   - the checkpoint with the lowest validation loss is saved to `best_model.pth`
5. **Inference on validation images**: draws predicted boxes with a confidence score ≥ 0.5.
6. **Custom images**: upload your own image (Google Colab upload widget) and run detection on it.
7. **Saving and loading**: rebuilds the model from scratch, loads `best_model.pth` and runs inference without retraining.

## Classes

`aeroplane`, `bicycle`, `bird`, `boat`, `bottle`, `bus`, `car`, `cat`, `chair`, `cow`, `diningtable`, `dog`, `horse`, `motorbike`, `person`, `pottedplant`, `sheep`, `sofa`, `train`, `tvmonitor`

## Results

Trained on a CUDA GPU in Google Colab. Losses are the summed Faster R-CNN losses (classifier, box regression, objectness, RPN box regression):

| Epoch | Train loss | Val loss |
|------:|-----------:|---------:|
| 1  | 0.4927 | 0.3509 |
| 3  | 0.2963 | 0.3134 |
| 5  | 0.2534 | 0.3118 |
| **7**  | **0.2031** | **0.2982** (best, saved) |
| 10 | 0.1935 | 0.3037 |
| 15 | 0.1877 | 0.3075 |

Validation loss was lowest at epoch 7 and then rose slightly while training loss kept falling. This means the model began to overfit, so the epoch 7 weights are the ones kept in `best_model.pth`.

> The notebook reports loss only. It doesn't compute mAP, the standard VOC detection metric.

## Getting started

### Option A: Google Colab (recommended)

The notebook was written for Colab. Open it there, choose **Runtime → Change runtime type → GPU**, and run all cells. The dataset (~900 MB) is downloaded into `./data` automatically.

### Option B: Run locally

```bash
pip install torch torchvision matplotlib numpy pillow jupyter
jupyter notebook PASCAL_VOC_2007.ipynb
```

Notes for local use:
- You need a recent torchvision (0.16 or later) for `transforms.v2` and `tv_tensors`.
- A GPU is strongly recommended. Training on a CPU will be very slow.
- The custom-image cell (Step 6) uses `google.colab.files.upload()`, which only works in Colab. Locally, load an image with `Image.open("path/to/image.jpg")` instead.
- In Step 7, change `image_path` to point to an image on your machine.

### Using the trained weights only

```python
import torch, torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 21)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu", weights_only=True))
model.eval()
```

Label indices start at 1 and follow the class order above (0 = background).

## Repository contents

| File | Description |
|------|-------------|
| `PASCAL_VOC_2007.ipynb` | The full pipeline: data, training, evaluation and inference |
| `best_model.pth` | Fine-tuned weights (~166 MB) |
| `images.jpeg`, `*.webp` | Example images used to test the model on real-world photos |

## Acknowledgements

- [PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) dataset: Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J. and Zisserman, A.
- [Faster R-CNN](https://arxiv.org/abs/1506.01497): Ren, S., He, K., Girshick, R. and Sun, J. (2015)
- [PyTorch / torchvision](https://pytorch.org/vision/stable/models.html) detection models

## Author

Walusimbi Uthman
