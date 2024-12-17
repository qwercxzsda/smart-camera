import asyncio
import logging
import unittest
from pathlib import Path

from PIL import Image

from image_analyzer.image_describer.image_describer import DummyImageDescriber, ImageDescribed
from image_analyzer.object_detector.object_detector import Detection, DummyObjectDetector, ImageObjectDetected

logger = logging.getLogger(__name__)


class TestObjectDetector(unittest.TestCase):
    def setUp(self):
        self.detector = DummyObjectDetector()
        self.resource_dir = Path(__file__).parent / "resources"
        self.img: Image.Image = Image.open(self.resource_dir / "img1.png")

    def test_preprocess(self):
        preprocessed_image: Image.Image = self.detector.preprocess(self.img)
        self.assertEqual(preprocessed_image.size, (300, 300))
        preprocessed_image.save(self.resource_dir / "tmp" / "test_preprocess.png")

    def test_detect(self):
        res: ImageObjectDetected = asyncio.run(self.detector.detect(self.img))
        self.assertIsInstance(res, ImageObjectDetected)
        self.assertEqual(res.image_detected.size, (300, 300))
        self.assertEqual(len(res.detections), 1)
        detection: Detection = res.detections[0]
        self.assertEqual(detection.box, (0.1, 0.1, 0.9, 0.9))
        self.assertEqual(detection.score, 0.9)
        self.assertEqual(detection.class_id, 0)
        self.assertEqual(detection.class_name, "dummy")
        self.assertEqual(self.img, res.image)
        res.image_detected.save(self.resource_dir / "tmp" / "test_detect.png")


class TestHailoObjectDetector(unittest.TestCase):
    def setUp(self):
        from image_analyzer.object_detector.hailo_object_detector import HailoObjectDetector
        self.detector = HailoObjectDetector()
        self.resource_dir = Path(__file__).parent / "resources"
        self.img: Image.Image = Image.open(self.resource_dir / "img1.png")

    def test_detect1(self):
        res: ImageObjectDetected = asyncio.run(self.detector.detect(self.img))
        self.assertIsInstance(res, ImageObjectDetected)
        print(res.detections)
        res.image_detected.save(self.resource_dir / "tmp" / "test_hailo_detect1.png")

    def test_detect2(self):
        res: ImageObjectDetected = asyncio.run(self.detector.detect(self.img))
        self.assertIsInstance(res, ImageObjectDetected)
        print(res.detections)
        res.image_detected.save(self.resource_dir / "tmp" / "test_hailo_detect2.png")


class TestImageDescriber(unittest.TestCase):
    def setUp(self):
        self.describer = DummyImageDescriber()
        self.detector = DummyObjectDetector()
        self.resource_dir = Path(__file__).parent / "resources"
        self.img: Image.Image = Image.open(self.resource_dir / "img1.png")
        self.img_detected: ImageObjectDetected = asyncio.run(self.detector.detect(self.img))

    def test_describe(self):
        description: ImageDescribed = asyncio.run(self.describer.describe(self.img_detected))
        self.assertIsInstance(description, ImageDescribed)
        self.assertEqual(description.description, "A dummy description")
        self.assertEqual(description.status, "success")


class TestOllamaImageDescriber(unittest.TestCase):
    def setUp(self):
        from image_analyzer.image_describer.ollama_image_describer import OllamaImageDescriber
        self.describer = OllamaImageDescriber()
        self.detector = DummyObjectDetector()
        self.resource_dir = Path(__file__).parent / "resources"
        self.img: Image.Image = Image.open(self.resource_dir / "img2.png")
        self.img_detected: ImageObjectDetected = asyncio.run(self.detector.detect(self.img))

    def test_describe(self):
        description: ImageDescribed = asyncio.run(self.describer.describe(self.img_detected))
        self.assertIsInstance(description, ImageDescribed)
        self.assertEqual(description.status, "success")
        print("\n--------------------")
        print(description)
        print("--------------------\n")


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    unittest.main()
