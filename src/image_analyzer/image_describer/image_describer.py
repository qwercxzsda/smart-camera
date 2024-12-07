import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import final, override

from PIL import Image

from image_analyzer.object_detector.object_detector import ImageObjectDetected

__all__ = ["ImageDescribed", "ImageDescriber", "DummyImageDescriber"]

logger = getLogger(__name__)


@dataclass(frozen=True)
class ImageDescribed:
    image: ImageObjectDetected
    description: str
    status: str
    time: float


class ImageDescriber(ABC):
    def __init__(self, max_w_h: int):
        self.max_w_h: int = max_w_h
        self.processing: bool = False

    @final
    def preprocess(self, image: Image.Image) -> Image.Image:
        img_w, img_h = image.size
        max_w_h = self.max_w_h
        scale: float = min(max_w_h / img_w, max_w_h / img_h)

        if scale > 1:
            return image

        new_img_w, new_img_h = int(img_w * scale), int(img_h * scale)
        return image.resize(
            (new_img_w, new_img_h), Image.Resampling.BICUBIC
        )

    @abstractmethod
    async def describe_image(self, image: ImageObjectDetected) -> str:
        pass

    @final
    async def describe(self, image: ImageObjectDetected) -> ImageDescribed:
        if self.processing:
            logger.info("Already processing an image")
            return ImageDescribed(image=image, description="", status="busy", time=-1.)

        self.processing = True
        time_s: float = time.time()

        description = await self.describe_image(image)

        self.processing = False
        time_delta: float = time.time() - time_s
        logger.info(f"Described the image in {time_delta:.2f} seconds, description: {description}")

        return ImageDescribed(
            image=image, description=description, status="success", time=time_delta
        )


class DummyImageDescriber(ImageDescriber):
    def __init__(self):
        super().__init__(max_w_h=128)

    @override
    async def describe_image(self, image: ImageObjectDetected) -> str:
        await asyncio.sleep(3)
        return "A dummy description"
