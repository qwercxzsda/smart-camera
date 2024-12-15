import io
import logging
import uuid
from pathlib import Path
from typing import Annotated, Optional

from PIL import Image
from fastapi import Cookie, FastAPI, Response, UploadFile
from fastapi.staticfiles import StaticFiles

from image_analyzer.image_analyzer import ImageAnalyzer
from image_analyzer.image_describer.image_describer import DummyImageDescriber, ImageDescribed, ImageDescriber
from image_analyzer.image_describer.ollama_image_describer import base64encode
from image_analyzer.object_detector.object_detector import DummyObjectDetector, ObjectDetector

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    filemode="w",
    filename="server.log",
)

current_dir: Path = Path(__file__).parent
static_dir: Path = current_dir / "client" / "dist"
logging.info(f"Starting server, serving static files from {static_dir}")

# Initialize the ImageAnalyzer
object_detector: ObjectDetector = DummyObjectDetector()
image_describer: ImageDescriber = DummyImageDescriber()
image_analyzer: ImageAnalyzer = ImageAnalyzer(object_detector, image_describer)

app = FastAPI()


@app.post("/api/analyze")
async def analyze(
        response: Response,
        file: UploadFile,
        user: Annotated[Optional[str], Cookie()] = None
):
    if not user:
        user = str(uuid.uuid4())
        response.set_cookie(key="user", value=user)
        logging.info(f"Created new user cookie: {user}")

    contents = await file.read()
    image: Image.Image = Image.open(io.BytesIO(contents))

    # Analyze the image
    image_described: ImageDescribed = await image_analyzer.analyze(user, image)

    return {
        "image": base64encode(image_described.image.image_detected),
        "status": image_described.status,
        "detections": str(image_described.image.detections),
        "description": image_described.description,
        "time": image_described.time,
    }


@app.post("/api/refresh")
async def refresh(
        response: Response,
        user: Annotated[Optional[str], Cookie()] = None
):
    if not user:
        return {"message": "No user cookie found."}
    response.delete_cookie("user")
    image_analyzer.refresh(user)
    return {"message": "User cookie deleted."}


app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
