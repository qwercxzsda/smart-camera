import base64
import io
import logging
from typing import Any, override

import aiohttp
from PIL import Image
from aiohttp import ClientResponse

from image_analyzer.image_describer.image_describer import ImageDescriber
from image_analyzer.object_detector.object_detector import ImageObjectDetected

__all__ = ["OllamaImageDescriber"]

logger = logging.getLogger(__name__)

ollama_port: int = 11434
ollama_addr: str = f"http://localhost:{ollama_port}/api/generate"
ollama_max_w_h: int = 672
ollama_model: str = "llava:7b"
ollama_prompt: str = "Assume you are in the photo. Briefly descibe the nearby surroundings."


def base64encode(image: Image.Image) -> str:
    """
    Encode an image as a base64 string.
    :param image: The image to encode.
    :return: The base64 string.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


async def query(image: Image.Image) -> str:
    """
    Query the OLLAMA server with an image.
    :param image: The image to query.
    :return: The response from the OLLAMA server.
    """
    async with aiohttp.ClientSession() as session:
        response: ClientResponse = await session.post(ollama_addr, json={
            "model": ollama_model,
            "prompt": ollama_prompt,
            "stream": False,
            "images": [base64encode(image)],
        })

    response_full: dict[str, Any] = await response.json()
    logger.info(f"Response from ollama: {response_full}")
    assert 'response' in response_full, (
        f"Invalid response_full from OLLAMA: {response_full}"
    )
    return response_full['response']


class OllamaImageDescriber(ImageDescriber):
    def __init__(self):
        super().__init__(ollama_max_w_h)

    @override
    async def describe_image(self, image: ImageObjectDetected) -> str:
        image_resized: Image.Image = self.preprocess(image.image)
        return await query(image_resized)
