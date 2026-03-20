This project implements prompt-based image segmentation using the CLIPSeg model.
The goal is to segment regions in images using natural language prompts.

Prompts used in this project:
- segment crack
- segment taping area

The model predicts segmentation masks corresponding to the given prompt and image.


MODEL

The project uses the CLIPSeg model from HuggingFace.

Model used:
CIDAS/clipseg-rd64-refined

CLIPSeg performs segmentation using the following pipeline:
Image + Text Prompt -> CLIP Encoder -> Segmentation Decoder -> Mask

It enables text-guided segmentation, allowing the model to detect regions in an image
based on natural language descriptions.


DATASET

Two datasets were used in this project.

1. Drywall Dataset
Task:
segment taping area

Annotation type:
Bounding Boxes

Bounding boxes are converted into binary masks for training.

2. Crack Dataset
Task:
segment crack

Annotation type:
Polygon segmentation

Polygon annotations are converted into binary masks for training.


DATASET SPLIT

Drywall dataset:
- Train: 90 percent
- Validation: 10 percent
- Test: provided validation/test split

Crack dataset:
- Train
- Validation
- Test

Both datasets are combined during training and evaluation.

Both training and the inference code is given in the same notebook

PROJECT STRUCTURE

project/
|
|-- dataloader.py
|-- Clipseg-training_and_inference_code.ipny
|
|-- predictions/
|-- overlays/
|-- triplet_visualizations/
|-- gt_vs_prediction/
|
|-- best_clipseg_model.pth
|-- clipseg_checkpoint.pth
|
|-- training_log.txt
|-- test_metrics.csv
|-- test_summary.txt
|-- model_stats.csv
|
|-- README.txt


TRAINING

Training uses the following configuration:

Optimizer:
AdamW

Learning Rate:
1e-5

Loss Function:
Weighted Binary Cross Entropy + Dice Loss

Scheduler:
ReduceLROnPlateau

The training pipeline supports checkpoint resuming.
If training stops, it automatically resumes from the last saved checkpoint.

During training the following files are generated:
- clipseg_checkpoint.pth
- best_clipseg_model.pth
- training_log.txt
- best_model_info.pkl


INFERENCE

The inference code is given with training code 
1. Load trained CLIPSeg model
2. Run segmentation using text prompts
3. Generate binary segmentation masks
4. Compute evaluation metrics
5. Save visualization outputs


OUTPUT FILES

1. Binary Masks
Saved in:
predictions/

Filename format:
imageID__prompt.png

Properties:
- PNG format
- single channel
- values {0,255}
- same spatial size as input image

2. Overlay Images
Saved in:
overlays/

These images show the segmentation mask overlaid on the original image.

3. Triplet Visualization
Saved in:
triplet_visualizations/

Each image contains:
Original Image | Predicted Mask | Overlay

4. Ground Truth vs Prediction
Saved in:
gt_vs_prediction/

Each image contains:
Ground Truth | Prediction

This allows easy visual comparison of model performance.


EVALUATION METRICS

Two evaluation metrics are used.

1. Dice Score
Dice measures the overlap between predicted mask and ground truth mask.


2. Mean Intersection over Union (mIoU)
IoU measures the intersection between prediction and ground truth divided by their union.

METRIC OUTPUT FILES

Per-image metrics are saved in:
test_metrics.csv