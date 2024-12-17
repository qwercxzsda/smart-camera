import './style.css';
import {analyze, AnalyzeResult} from './analyze';
import {VideoManager} from './video-manager';
import {API_REFRESH, MAX_ELEMENTS, UPDATE_INTERVAL} from "./config.ts";


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

    const refreshButton: HTMLButtonElement | null = document.querySelector<HTMLButtonElement>('#refresh-button');
    console.assert(refreshButton !== null, 'refreshButton is null');
    setupRefreshButton(refreshButton!, outputList!);
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

function setupRefreshButton(button: HTMLButtonElement, outputList: HTMLUListElement): void {
    button.addEventListener('click', async function () {
        await fetch(API_REFRESH, {
            method: 'POST',
            credentials: 'include'
        });

        // Refresh outputList by clearing its children
        Array.from(outputList.children).forEach(function (child: Element): void {
            const imgElement: HTMLImageElement | null = child.querySelector('img');
            if (imgElement && imgElement.src !== '/ollama.png') {
                URL.revokeObjectURL(imgElement.src); // Revoke URL before removing
            }
        });
        outputList.innerHTML = ''; // Clear all children from the output list
    });
}

function setupAnalyze(
    videoManager: VideoManager,
    detectedImage: HTMLImageElement, detectedText: HTMLParagraphElement,
    outputList: HTMLUListElement
): void {
    setInterval(async function () {
        await analyzeAndUpdate(videoManager, detectedImage, detectedText, outputList);
    }, UPDATE_INTERVAL);
}

async function analyzeAndUpdate(
    videoManager: VideoManager,
    detectedImage: HTMLImageElement, detectedText: HTMLParagraphElement,
    outputList: HTMLUListElement
): Promise<void> {
    const image: File = await videoManager.getVideoFrameUnsafe();
    const analyzeResult: AnalyzeResult = await analyze(image);

    if (detectedImage.src !== '/hailo.png') {
        URL.revokeObjectURL(detectedImage.src);
    }
    detectedImage.src = URL.createObjectURL(analyzeResult.image);
    detectedText.textContent = (
        `status: ${analyzeResult.status}, took ${analyzeResult.time.toFixed(4)} seconds\n\n${analyzeResult.detections}`
    );

    if (analyzeResult.status !== 'success') {
        console.log(`status: ${analyzeResult.status}, skipping output list update`);
        return;
    }
    console.log(`status: ${analyzeResult.status}, updating output list`);
    if (outputList.children.length > MAX_ELEMENTS) {
        const lastChild: Element = outputList.children[outputList.children.length - 1];
        const imgElement: HTMLImageElement | null = lastChild.querySelector('img');
        if (imgElement && imgElement.src !== '/ollama.png') {
            URL.revokeObjectURL(imgElement.src);
        }
        outputList.removeChild(lastChild);
    }

    const li: HTMLLIElement = document.createElement('li');
    li.classList.add('output-item');

    const img: HTMLImageElement = document.createElement('img');
    img.src = URL.createObjectURL(analyzeResult.image);
    img.alt = 'Output Image';
    img.classList.add('output-image');

    const p: HTMLParagraphElement = document.createElement('p');
    p.classList.add('output-text');
    p.textContent = `took ${analyzeResult.time.toFixed(2)} seconds\n${analyzeResult.description}`;

    li.appendChild(img);
    li.appendChild(p);
    outputList.insertBefore(li, outputList.firstChild);
}
