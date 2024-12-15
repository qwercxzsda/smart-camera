import asyncio
import threading
from logging import getLogger
from pathlib import Path
from queue import Queue
from typing import Any, Optional

import numpy as np
from PIL import Image

from image_analyzer.object_detector.hailo_async_interface import HailoAsyncInference
from image_analyzer.object_detector.object_detector import Detection, ObjectDetector

__all__ = ["HailoObjectDetector"]

logger = getLogger(__name__)


def get_labels(labels_path: Path) -> list[str]:
    """
    Load labels from a file.

    Args:
        labels_path (str): Path to the labels file.

    Returns:
        list: List of class names.
    """
    with open(labels_path, 'r', encoding="utf-8") as f:
        class_names = f.read().splitlines()
    return class_names


class HailoObjectDetector(ObjectDetector):
    def __init__(self):
        self.model_path: Path = (Path(__file__).parent / "yolov10b.hef").resolve()
        self.w, self.h = 640, 640
        self.threshold: float = 0.5
        super().__init__(self.w, self.h)

        self.labels_path: Path = (Path(__file__).parent / "coco.txt").resolve()
        self.labels: list[str] = get_labels(self.labels_path)

        # We only allow one item in the queue at a time
        # This is to ensure that the outputs are not mixed up
        self.lock: asyncio.Lock = asyncio.Lock()
        self.input_queue: Queue[Optional[list[Image.Image]]] = Queue()
        self.output_queue: Queue[tuple[Any, list[Any]]] = Queue()
        hailo_async_inference: HailoAsyncInference = HailoAsyncInference(
            str(self.model_path),
            self.input_queue, self.output_queue, batch_size=1
        )
        threading.Thread(target=hailo_async_inference.run, daemon=True).start()

    def extract_detections(self, outputs: list[np.ndarray]) -> list[Detection]:
        detections: list[Detection] = []
        for i, output in enumerate(outputs):
            assert len(output.shape) == 2, f"Expected 2 dimensions in output, got {output.shape}"
            for det in output:
                bbox, score = det[:4].tolist(), det[4].item()
                if score < self.threshold:
                    continue

                assert len(bbox) == 4, f"Expected 4 values in bbox, got {bbox}"
                bbox_t: tuple[float, float, float, float] = tuple(bbox)
                detections.append(Detection(
                    box=bbox_t,
                    score=score,
                    class_id=i,
                    class_name=self.labels[i]
                ))

        return detections

    async def run(self, image_preprocessed: Image.Image) -> list[Detection]:
        async with self.lock:
            self.input_queue.put([image_preprocessed])
            # We put a single image, so we expect a single output
            _, outputs = await asyncio.to_thread(self.output_queue.get)

        # output may be list[list[np.ndarray]] (hailort versions < 4.19.0) or list[np.ndarray]
        # In both cases, the inner list has len(classes) elements
        if len(outputs) == 1:
            outputs = outputs[0]
        return self.extract_detections(outputs)

    async def detect_objects(self, image_preprocessed: Image.Image) -> list[Detection]:
        return await self.run(image_preprocessed)
