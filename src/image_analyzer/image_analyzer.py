import time
from collections import defaultdict
from logging import getLogger
from typing import Any, Optional

from PIL import Image

from image_analyzer.image_describer.image_describer import ImageDescribed, ImageDescriber
from image_analyzer.object_detector.object_detector import ImageObjectDetected, ObjectDetector

__all__ = ["ImageAnalyzer"]

logger = getLogger(__name__)


def get_last_element(lst: list[Any]) -> Optional[Any]:
    """
    Get the last element of a list, or None if the list is empty.
    """
    return lst[-1] if lst else None


def is_different(image1: ImageObjectDetected, image2: Optional[ImageObjectDetected]) -> bool:
    """
    Check if two images are different in terms of the detected objects.
    """
    if image2 is None:
        return True

    image1_detections: defaultdict[str, int] = defaultdict(int)
    image2_detections: defaultdict[str, int] = defaultdict(int)

    for detection in image1.detections:
        image1_detections[detection.class_name] += 1
    for detection in image2.detections:
        image2_detections[detection.class_name] += 1

    return set(image1_detections.items()) != set(image2_detections.items())


class ImageAnalyzer:
    def __init__(self, object_detector: ObjectDetector, image_describer: ImageDescriber):
        self.object_detector = object_detector
        self.image_describer = image_describer

        self.history: defaultdict[str, list[ImageObjectDetected]] = defaultdict(list)

    async def analyze_image(self, user: str, image_raw: Image.Image) -> ImageDescribed:
        logger.info(f"Analyzing image for user {user}, user history length: {len(self.history[user])}")
        time_s: float = time.time()
        image: ImageObjectDetected = await self.object_detector.detect(image_raw)

        prev_image: Optional[ImageObjectDetected] = get_last_element(self.history[user])

        if not is_different(image, prev_image):
            logger.info(f"Image is the same as the previous image for user {user}")
            return ImageDescribed(
                image=image, description="", status="indifferent", time=time.time() - time_s
            )

        self.history[user].append(image)
        image_described: ImageDescribed = await self.image_describer.describe(image)
        return ImageDescribed(
            image=image,
            description=image_described.description, status=image_described.status,
            time=time.time() - time_s
        )

    async def analyze(self, user: str, image_raw: Image.Image) -> ImageDescribed:
        image_described: ImageDescribed = await self.analyze_image(user, image_raw)
        logger.info(f"image_described: {image_described}")
        assert image_described.status in ["success", "indifferent", "busy"], (
            f"Unexpected status: {image_described.status}"
        )

        return image_described

    def refresh(self, user: str) -> None:
        if user not in self.history:
            logger.info(f"User {user} not found in history")
        else:
            logger.info(f"Deleting user history of length {len(self.history[user])} for user {user}")
            self.history.pop(user)
