export interface AnalyzeResult {
    image: File;
    status: string;
    detections: string;
    description: string;
}

let counter: number = 0;

// TODO: fetch from server
export async function analyze(image: File): Promise<AnalyzeResult> {
    if (counter++ % 50 !== 0) {
        return {
            image,
            status: 'busy',
            detections: 'nothing!',
            description: 'I dont know...'
        };
    }
    return {
        image,
        status: 'success',
        detections: 'something?',
        description: "It's a nice image"
    };
}