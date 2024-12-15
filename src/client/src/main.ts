import './style.css';
import {analyze, AnalyzeResult} from './analyze';
import {VideoManager} from './video-manager';

const updateInterval: number = 5000;
const maxElements: number = 50;

main();

function main() {
    const cameraFeed: HTMLVideoElement | null = document.querySelector<HTMLVideoElement>('#camera-feed');
    console.assert(cameraFeed !== null, 'cameraFeed is null');
    setupCameraFeed(cameraFeed!);

    const videoManager = new VideoManager(cameraFeed!);
    const detectedImage: HTMLImageElement | null = document.querySelector<HTMLImageElement>('#detected-image');
    console.assert(detectedImage !== null, 'detectedImage is null');
    const detectedText: HTMLParagraphElement | null = document.querySelector<HTMLParagraphElement>('#detected-text');
    console.assert(detectedText !== null, 'detectedText is null');
    const outputList: HTMLUListElement | null = document.querySelector<HTMLUListElement>('#output-list');
    console.assert(outputList !== null, 'outputList is null');
    setupAnalyze(videoManager, detectedImage!, detectedText!, outputList!);
}

function setupCameraFeed(video: HTMLVideoElement): void {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({video: true})
            .then(stream => {
                video.srcObject = stream;
            })
            .catch(err => {
                console.error("Error accessing the camera: ", err);
            });
    } else {
        console.error("Browser doesn't support accessing the camera.");
    }
}

function setupAnalyze(
    videoManager: VideoManager,
    detectedImage: HTMLImageElement, detectedText: HTMLParagraphElement,
    outputList: HTMLUListElement
): void {
    setInterval(async function () {
        await analyzeAndUpdate(videoManager, detectedImage, detectedText, outputList);
    }, updateInterval);
}

async function analyzeAndUpdate(
    videoManager: VideoManager,
    detectedImage: HTMLImageElement, detectedText: HTMLParagraphElement,
    outputList: HTMLUListElement
): Promise<void> {
    const image: File = await videoManager.getVideoFrameUnsafe();
    const analyzeResult: AnalyzeResult = await analyze(image);
    detectedImage.src = URL.createObjectURL(analyzeResult.image);
    detectedText.textContent = `status: ${analyzeResult.status}\n\n${analyzeResult.detections}`;

    if (analyzeResult.status !== 'success') {
        console.log(`status: ${analyzeResult.status}, skipping output list update`);
        return;
    }
    console.log(`status: ${analyzeResult.status}, updating output list`);
    if (outputList.children.length > maxElements) {
        outputList.removeChild(outputList.children[outputList.children.length - 1]);
    }

    const li: HTMLLIElement = document.createElement('li');
    li.classList.add('output-item');

    const img: HTMLImageElement = document.createElement('img');
    img.src = URL.createObjectURL(analyzeResult.image);
    img.alt = 'Output Image';
    img.classList.add('output_image');

    const p: HTMLParagraphElement = document.createElement('p');
    p.classList.add('output-text');
    p.textContent = `took ${analyzeResult.time.toFixed(2)} seconds\n${analyzeResult.description}`;

    li.appendChild(img);
    li.appendChild(p);
    outputList.insertBefore(li, outputList.firstChild);
}
