import os

import defusedxml.ElementTree as ET
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class LandmarksDataset(Dataset):
    def __init__(self, data_dir, xml_file_path, transforms=None) -> None:
        self.data_dir: str = data_dir
        self.samples: list[dict] = self.load_data(os.path.join(self.data_dir, xml_file_path))
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        sample: dict = self.samples[index]

        file_name = sample["file"]
        img_path = os.path.join(self.data_dir, file_name)

        # Get image
        img = Image.open(img_path).convert("RGB")

        # Get bounding box
        box_left = sample["box_left"]
        box_top = sample["box_top"]
        box_width = sample["box_width"]
        box_height = sample["box_height"]

        # Crop image
        img = img.crop((box_left, box_top, box_left + box_width, box_top + box_height))
        img = np.array(img)

        # Normalize landmarks
        landmarks = np.array(sample["landmarks"]) - np.array([box_left, box_top])

        if self.transforms:
            transformed = self.transforms(image=img, keypoints=landmarks)
            img = transformed["image"]
            landmarks = transformed["keypoints"]
            _, height, width = img.shape
            landmarks /= np.array([width, height])
            landmarks -= 0.5

        return img, torch.Tensor(landmarks)

    @staticmethod
    def annotate_landmarks(
        img: torch.Tensor, landmarks: torch.Tensor, is_ground_truth: bool = True
    ) -> torch.Tensor:
        """Annotate landmarks on image

        Args:
            img (torch.Tensor):
            landmarks (torch.Tensor): normalized landmarks [-0.5, 0.5]

        Returns:
            torch.Tensor: _description_
        """
        img = img.cpu().clone()
        landmarks = landmarks.cpu().clone()

        _, height, width = img.shape
        landmarks += 0.5
        landmarks *= np.array([width, height])
        img = img.permute(1, 2, 0).numpy()
        img = Image.fromarray((img * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)

        # Set color
        if is_ground_truth:
            color = (0, 255, 0)  # Green for ground truth
        else:
            color = (255, 0, 0)  # Red for predict
        # Draw landmarks
        for x, y in landmarks:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)

        # Convert to torch.Tensor
        img = np.array(img).astype(np.float32) / 255.0
        img = ToTensorV2()(image=img)["image"]
        return img

    def load_data(self, xml_file_path: str) -> list[dict]:
        """Load data: file_path, bbox, landmarks

        Args:
            xml_file_path (str): xml file path

        Returns:
            list[dict]: list[{
                "file_name": str,
                "width": int,
                "height": int,
                "box_top": int,
                "box_left": int,
                "box_width": int,
                "box_height": int,
                "landmarks": np.array()
            }]
        """
        images = ET.parse(xml_file_path).getroot().find("images")
        return [self.parse_image(image) for image in images]

    def parse_image(self, image: ET) -> dict:
        """Parse ET.ElementTree to dict.

        Args:
            image (ET.ElementTree): ET.ElementTree

        Returns:
            dict: {
                "file": str,
                "width": int,
                "height": int,
                "box_top": int,
                "box_left": int,
                "box_width": int,
                "box_height": int,
                "landmarks": np.array()
            }
        """
        file = image.attrib["file"]
        width = int(image.attrib["width"])
        height = int(image.attrib["height"])

        box = image.find("box")
        box_top = int(box.attrib["top"])
        box_left = int(box.attrib["left"])
        box_width = int(box.attrib["width"])
        box_height = int(box.attrib["height"])

        landmarks = np.array([[float(part.attrib["x"]), float(part.attrib["y"])] for part in box])

        return dict(
            file=file,
            width=width,
            height=height,
            box_top=box_top,
            box_left=box_left,
            box_width=box_width,
            box_height=box_height,
            landmarks=landmarks,
        )
