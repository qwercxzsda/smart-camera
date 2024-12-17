import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from typing import final

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from numpy.random import default_rng

__all__ = ["Detection", "ImageObjectDetected", "ObjectDetector", "DummyObjectDetector", "detections_to_str"]

logger = getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class ImageObjectDetected:
    image: Image.Image
    image_detected: Image.Image
    detections: list[Detection]


def detections_to_str(detections: list[Detection]) -> str:
    detections_dict: defaultdict[str, int] = defaultdict(int)
    for detection in detections:
        detections_dict[detection.class_name] += 1

    return "\n".join([f"{class_name}: {count}" for class_name, count in detections_dict.items()])


def class_id_to_color(class_id: int) -> tuple[int, int, int]:
    generator: np.random.Generator = default_rng(class_id)
    color: list[int] = generator.integers(0, 256, size=3).tolist()
    assert len(color) == 3, f"Expected 3 values in color, got {color}"
    r, g, b = color
    return r, g, b


def draw_detection(
        draw: ImageDraw.Draw, detection: Detection,
        color: tuple[int, int, int], height: int, width: int
) -> None:
    label: str = f"{detection.class_name}: {detection.score:.2f}%"
    ymin, xmin, ymax, xmax = detection.box
    ymin, xmin, ymax, xmax = ymin * height, xmin * width, ymax * height, xmax * width

    font = ImageFont.load_default(size=15)
    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=2)
    draw.text((xmin + 4, ymin + 4), label, fill=color, font=font)


def draw_detections(image_detected: Image.Image, detections: list[Detection]) -> None:
    draw: ImageDraw.Draw = ImageDraw.Draw(image_detected)
    height, width = image_detected.size
    for detection in detections:
        draw_detection(
            draw, detection,
            class_id_to_color(detection.class_id), height, width
        )


class ObjectDetector(ABC):
    def __init__(self, preprocess_width: int, preprocess_height: int):
        self.preprocess_width: int = preprocess_width
        self.preprocess_height: int = preprocess_height
        self.padding_color: tuple[int, int, int] = (114, 114, 114)

    @final
    def preprocess(self, image: Image.Image) -> Image.Image:
        img_w, img_h = image.size
        p_w, p_h = self.preprocess_width, self.preprocess_height
        scale: float = min(p_w / img_w, p_h / img_h)
        new_img_w, new_img_h = int(img_w * scale), int(img_h * scale)

        image_resized: Image.Image = image.resize(
            (new_img_w, new_img_h), Image.Resampling.BICUBIC
        )

        padded_image = Image.new('RGB', (p_w, p_h), self.padding_color)
        padded_image.paste(image_resized, ((p_w - new_img_w) // 2, (p_h - new_img_h) // 2))
        return padded_image

    @abstractmethod
    async def detect_objects(self, image_preprocessed: Image.Image) -> list[Detection]:
        pass

    @final
    async def detect(self, image: Image.Image) -> ImageObjectDetected:
        image_detected: Image.Image = self.preprocess(image)
        detections: list[Detection] = await self.detect_objects(image_detected)
        draw_detections(image_detected, detections)
        logger.info(f"Detected {len(detections)} objects in the image, detections: {detections}")

        return ImageObjectDetected(
            image=image, image_detected=image_detected, detections=detections
        )


class DummyObjectDetector(ObjectDetector):
    def __init__(self):
        super().__init__(preprocess_width=300, preprocess_height=300)

    async def detect_objects(self, image_preprocessed: Image.Image) -> list[Detection]:
        await asyncio.sleep(0.1)
        return [
            Detection(
                box=(0.1, 0.1, 0.9, 0.9),
                score=0.9,
                class_id=0,
                class_name="dummy"
            )
        ]
