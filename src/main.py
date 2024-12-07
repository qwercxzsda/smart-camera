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
from image_analyzer.object_detector.object_detector import DummyObjectDetector, ObjectDetector

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)

current_dir: Path = Path(__file__).parent
static_dir: Path = current_dir / "client" / "dist"
logging.debug(f"Starting server, serving static files from {static_dir}")

# Initialize the ImageAnalyzer
object_detector: ObjectDetector = DummyObjectDetector()
image_describer: ImageDescriber = DummyImageDescriber()
image_analyzer: ImageAnalyzer = ImageAnalyzer(object_detector, image_describer)

app = FastAPI()
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


@app.post("/analyze")
async def analyze(
        response: Response,
        file: UploadFile,
        user: Annotated[Optional[str], Cookie()] = None
):
    if not user:
        user = str(uuid.uuid4())
        response.set_cookie(key="user", value=user)
        logging.debug(f"Created new user cookie: {user}")

    contents = await file.read()
    image: Image.Image = Image.open(io.BytesIO(contents))

    # Analyze the image
    image_described: ImageDescribed = await image_analyzer.analyze(user, image)

    return {
        "filename": file.filename,
        "description": image_described.description,
        "image_detected": image_described.image.image_detected,
        "size": image_described.image.image_detected.size,
        "format": image_described.image.image_detected.format,
    }


@app.post("/refresh")
async def refresh(
        response: Response,
        user: Annotated[Optional[str], Cookie()] = None
):
    if not user:
        return {"message": "No user cookie found."}
    response.delete_cookie("user")
    image_analyzer.refresh(user)
    return {"message": "User cookie deleted."}
