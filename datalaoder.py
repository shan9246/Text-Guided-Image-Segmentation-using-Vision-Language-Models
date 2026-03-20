import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from collections import defaultdict


class COCODataset(Dataset):

    def __init__(self,
                 root_dir,
                 annotation_file,
                 annotation_type="polygon",
                 prompt="segment object"):

        self.root_dir = root_dir
        self.annotation_type = annotation_type
        self.prompt = prompt

        with open(annotation_file) as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]

        self.img_to_anns = defaultdict(list)

        for ann in self.annotations:
            self.img_to_anns[ann["image_id"]].append(ann)

        # Image transform
        self.img_transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])

        # Mask resize
        self.mask_resize = transforms.Resize(
            (640, 640),
            interpolation=Image.NEAREST
        )

    def __len__(self):
        return len(self.images)

    def polygon_to_mask(self, polygons, height, width):

        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        for polygon in polygons:
            xy = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
            draw.polygon(xy, outline=1, fill=1)

        return np.array(mask)

    def bbox_to_mask(self, bbox, height, width):

        x, y, w, h = bbox

        mask = np.zeros((height, width), dtype=np.uint8)

        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)

        mask[y1:y2, x1:x2] = 1

        return mask

    def __getitem__(self, idx):

        img_info = self.images[idx]

        img_id = img_info["id"]
        img_name = img_info["file_name"]

        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        width, height = image.size

        anns = self.img_to_anns[img_id]

        mask = np.zeros((height, width), dtype=np.uint8)

        for ann in anns:

            if self.annotation_type == "polygon" and "segmentation" in ann:

                poly_mask = self.polygon_to_mask(
                    ann["segmentation"],
                    height,
                    width
                )

                mask = np.maximum(mask, poly_mask)

            elif self.annotation_type == "bbox":

                bbox_mask = self.bbox_to_mask(
                    ann["bbox"],
                    height,
                    width
                )

                mask = np.maximum(mask, bbox_mask)

        mask = Image.fromarray(mask)

        # Resize image
        image = self.img_transform(image)

        # Resize mask
        mask = self.mask_resize(mask)

        mask = np.array(mask)

        # Ensure binary
        mask = (mask > 0).astype(np.float32)

        mask = torch.tensor(mask).unsqueeze(0)

        target = {
            "mask": mask,
            "prompt": self.prompt,
            "image_id": img_id,
            "file_name": img_name
        }

        return image, target