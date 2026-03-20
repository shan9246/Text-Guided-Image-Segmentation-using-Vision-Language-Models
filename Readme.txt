Project Title:
Prompt-Conditioned Segmentation for Drywall Quality Assessment

------------------------------------------------------------

1. Project Goal
------------------------------------------------------------

The objective of this project is to train and fine-tune a text-conditioned image segmentation model that produces a binary mask given an input image and a natural language prompt.

The model is required to segment two types of regions:

1. "segment crack" (Cracks dataset)
2. "segment taping area" (Drywall-Join-Detect dataset)

The system takes an image and a text prompt as input and outputs a binary segmentation mask corresponding to the prompt.

------------------------------------------------------------

2. Models Used
------------------------------------------------------------

Two different models were explored in this project:

1. CLIPSeg
2. CLIP + SegFormer


The second approach combines CLIP with the SegFormer transformer segmentation backbone. In this approach:

- SegFormer extracts hierarchical spatial features from the image
- CLIP encodes the text prompt
- The text embedding is fused with the visual features
- A decoder predicts the final segmentation mask

project Structure/
|
|-- dataloader.py
|-- segformer_clip_training_inference_code.ipny
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


------------------------------------------------------------

3. Dataset Description
------------------------------------------------------------

Two datasets were used:

1. Cracks Dataset
   - Contains images of surface cracks
   - Used with the prompt: "segment crack"

2. Drywall-Join-Detect Dataset
   - Contains images of drywall taping areas
   - Used with the prompt: "segment taping area"

------------------------------------------------------------

4. Dataset Split
------------------------------------------------------------

Drywall-Join-Detect
Train: 738
Validation: 82
Test: 202

Cracks
Train: 5164
Validation: 201
Test: 4

Combined Dataset
Train: 5902
Validation: 283
Test: 206

------------------------------------------------------------

5. Training Details
------------------------------------------------------------

Training configuration:

Epochs: 101
Batch size: 16
Optimizer: Adam
Learning rate scheduler: ReduceLROnPlateau
Scheduler factor: 0.5
Scheduler patience: 15

Loss function:

Combined loss was used:
- Weighted Binary Cross Entropy Loss
- Dice Loss

This combination helps address the strong class imbalance between foreground pixels (cracks or taping areas) and background pixels.

------------------------------------------------------------

6. Evaluation Metrics
------------------------------------------------------------

The models were evaluated using:

1. Dice Score
2. Intersection over Union (IoU)

Both metrics measure the overlap between predicted segmentation masks and ground truth masks.

------------------------------------------------------------

7. Output Results
------------------------------------------------------------

The inference pipeline produces:

- Predicted segmentation masks
- Overlay visualizations
- Ground truth vs prediction comparisons

Outputs are stored in the following folders:

predictions/
overlays/
visualizations/
gt_vs_pred/

Additional files generated:

test_metrics.csv
test_summary.txt
model_stats.csv

------------------------------------------------------------

8. Runtime Information
------------------------------------------------------------

The inference code records:

- Model size
- Average inference time per image
- Total inference time

These statistics are stored in:

model_stats.csv

------------------------------------------------------------

9. Files Included
------------------------------------------------------------

dataloader.py
training and inference script
trained model weights
README.txt
seeded_note.txt

------------------------------------------------------------

10. How to Run
------------------------------------------------------------

1. Training and inference 

Just run the segformer_clip_training_inference_code.py



Outputs will be saved in the results directories.

------------------------------------------------------------