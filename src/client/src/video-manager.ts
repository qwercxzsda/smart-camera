export class VideoManager {
    private frame: number;
    private readonly video: HTMLVideoElement;

    constructor(video: HTMLVideoElement) {
        this.frame = 0;
        this.video = video;
    }

    async getVideoFrame(): Promise<File | null> {
        const video: HTMLVideoElement = this.video;
        const canvas: HTMLCanvasElement = document.createElement('canvas');

        // Calculate aspect ratio and adjust canvas dimensions while maintaining the ratio
        const aspectRatio = video.videoWidth / video.videoHeight;
        if (aspectRatio > 1) {
            canvas.width = 640;
            canvas.height = 640 / aspectRatio;
        } else {
            canvas.width = 640 * aspectRatio;
            canvas.height = 640;
        }

        const context: CanvasRenderingContext2D | null = canvas.getContext('2d');
        if (context === null) {
            console.error("Failed to get canvas 2D context.");
            return null;
        }

        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob: Blob | null = await new Promise<Blob | null>(resolve =>
            canvas.toBlob(blob => resolve(blob), 'image/png')
        );
        if (blob === null) {
            console.error("Failed to capture image frame.");
            return null;
        }

        return new File([blob], `frame${this.frame++}.png`, {type: 'image/png'});
    }

    async getVideoFrameUnsafe(): Promise<File> {
        const videoFrame: File | null = await this.getVideoFrame();
        return videoFrame!
    }
}