import {API_ANALYZE} from "./config.ts";

export interface AnalyzeResult {
    image: File;
    status: string;
    detections: string;
    description: string;
    time: number;
}

let imageCounter: number = 0;

export async function analyze(image: File): Promise<AnalyzeResult> {
    console.log(`POST request to: ${API_ANALYZE}`);
    const formData = new FormData();
    formData.append('file', image);

    const response: Response = await fetch(API_ANALYZE, {
        method: 'POST',
        body: formData,
        credentials: 'include'
    });
    console.assert(response.ok, `Failed to analyze image, response status: ${response.status}`);

    const result = await response.json();
    console.assert(
        ['success', 'indifferent', 'busy'].includes(result.status),
        `Invalid status: ${result.status}`
    );

    const decodedImageBlob: Blob = await fetch(`data:image/png;base64,${result.image}`)
        .then(res => res.blob());
    const decodedImageFile = new File(
        [decodedImageBlob], `decodedImage${imageCounter++}.png`, {type: 'image/png'}
    );

    console.log(
        `status: ${result.status}, detections: ${result.detections}, 
        description: ${result.description}, time: ${result.time}`
    );

    return {
        image: decodedImageFile,
        status: result.status,
        detections: result.detections,
        description: result.description,
        time: result.time
    };
}