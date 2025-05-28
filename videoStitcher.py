import cv2
import imageTransformer

class VideoStitcher:
    def __init__(self,
            leftVideo: str,
            middleVideo: str,
            rightVideo: str,
            leftAngle: int,
            rightAngle: int,
            cameraFocalHeight: float,
            cameraFocalLength: float,
            projectionPlaneDistanceFromCenter: float,
            imageDimensions: (int, int)
    ):
        self.leftVideo = leftVideo
        self.middleVideo = middleVideo
        self.rightVideo = rightVideo
        self.leftAngle = leftAngle
        self.rightAngle = rightAngle
        self.cameraFocalHeight = cameraFocalHeight
        self.cameraFocalLength = cameraFocalLength
        self.projectionPlaneDistanceFromCenter = projectionPlaneDistanceFromCenter
        self.imageDimensions = imageDimensions

    def getCommonFrameRate (self):
        """Get a common frame rate from input videos."""
        cap_left = cv2.VideoCapture(self.leftVideo)
        fps_left = cap_left.get(cv2.CAP_PROP_FPS)
        cap_left.release()

        # If fps is unreliable, use a standard rate
        if fps_left <= 0:
            return 30.0
        return fps_left

    def _calculateOverlapRegions(self, leftFrame, middleFrame, rightFrame, transformationMatrices):
        """
        Calculate the overlap regions from the first frames.
        Returns the left-middle and middle-right overlap widths.
        """
        # Create transformer for these frames
        transformer = imageTransformer.ImageTransformer(
            leftImage=leftFrame,
            middleImage=middleFrame,
            rightImage=rightFrame,
            leftAngle=self.leftAngle,
            rightAngle=self.rightAngle,
            cameraFocalHeight=self.cameraFocalHeight,
            cameraFocalLength=self.cameraFocalLength,
            projectionPlaneDistanceFromCenter=self.projectionPlaneDistanceFromCenter,
            imageDimensions=self.imageDimensions,
            transformationMatrices=transformationMatrices
        )

        # Transform the images
        transformer.transformLeftImage()
        transformer.transformMiddleImage()
        transformer.transformRightImage()

        # Calculate overlap regions
        leftMiddleOverlap = transformer._findOverlapRegion(transformer.leftImage, transformer.middleImage, "right")
        middleRightOverlap = transformer._findOverlapRegion(transformer.middleImage, transformer.rightImage, "left")

        return leftMiddleOverlap, middleRightOverlap

    def _processFrame(self, leftFrame, middleFrame, rightFrame, transformationMatrices,
                      frameNumber=None, fps=None, precomputedOverlap=None):
        """
        Process a single set of frames from the three cameras.

        Args:
            leftFrame: Frame from the left camera
            middleFrame: Frame from the middle camera
            rightFrame: Frame from the right camera
            transformationMatrices: Pre-computed transformation matrices
            frameNumber: Optional frame number for timestamp overlay
            fps: Optional frames per second for timestamp calculation
            precomputedOverlap: Tuple of (leftMiddleOverlap, middleRightOverlap)

        Returns:
            The transformed and stitched frame
        """
        # Create transformer for these frames
        transformer = imageTransformer.ImageTransformer(
            leftImage=leftFrame,
            middleImage=middleFrame,
            rightImage=rightFrame,
            leftAngle=self.leftAngle,
            rightAngle=self.rightAngle,
            cameraFocalHeight=self.cameraFocalHeight,
            cameraFocalLength=self.cameraFocalLength,
            projectionPlaneDistanceFromCenter=self.projectionPlaneDistanceFromCenter,
            imageDimensions=self.imageDimensions,
            transformationMatrices=transformationMatrices
        )

        # Transform the images
        transformer.transformLeftImage()
        transformer.transformMiddleImage()
        transformer.transformRightImage()

        # Stitch the transformed images using pre-computed overlap if available
        stitchedFrame = transformer.stitchImages(precomputedOverlap)

        # Add timestamp overlay if frame number and fps are provided
        if frameNumber is not None and fps is not None:
            timestamp = frameNumber / fps
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            milliseconds = int((timestamp % 1) * 1000)

            # Add timestamp overlay
            time_text = f"Frame: {frameNumber} | Time: {minutes:02d}:{seconds:02d}.{milliseconds:03d}"
            cv2.putText(
                stitchedFrame,
                time_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Add FPS information
            cv2.putText(
                stitchedFrame,
                f"FPS: {fps:.2f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        return stitchedFrame

    def outputStitchedVideo(self, fileName: str, outputDir: str = "Outputs"):
        import os
        import numpy as np
        from subprocess import Popen, PIPE

        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        outputPath = os.path.join(outputDir, f"{fileName}.mp4")

        # Initialize video captures
        leftCap = cv2.VideoCapture(self.leftVideo)
        middleCap = cv2.VideoCapture(self.middleVideo)
        rightCap = cv2.VideoCapture(self.rightVideo)

        # Get frame rate
        fps = self.getCommonFrameRate()
        print(f"Source video frame rate: {fps} fps")

        # Get first frames to calculate dimensions
        leftRet, leftFrame = leftCap.read()
        middleRet, middleFrame = middleCap.read()
        rightRet, rightFrame = rightCap.read()

        if not (leftRet and middleRet and rightRet):
            print("Error: Could not read frames from one or more videos.")
            return

        # Create transformer and initialize matrices
        firstTransformer = imageTransformer.ImageTransformer(
            leftImage=leftFrame,
            middleImage=middleFrame,
            rightImage=rightFrame,
            leftAngle=self.leftAngle,
            rightAngle=self.rightAngle,
            cameraFocalHeight=self.cameraFocalHeight,
            cameraFocalLength=self.cameraFocalLength,
            projectionPlaneDistanceFromCenter=self.projectionPlaneDistanceFromCenter,
            imageDimensions=self.imageDimensions
        )

        transformationMatrices = firstTransformer.initializeTransformationMatrices()

        # Pre-compute overlap regions from first frame
        precomputedOverlap = self._calculateOverlapRegions(
            leftFrame, middleFrame, rightFrame, transformationMatrices)
        print(f"Pre-computed overlap regions: {precomputedOverlap}")

        # Process first frame to get dimensions
        firstFrame = self._processFrame(
            leftFrame, middleFrame, rightFrame,
            transformationMatrices,
            precomputedOverlap=precomputedOverlap
        )

        # Get original dimensions
        originalHeight, originalWidth = firstFrame.shape[:2]
        print(f"Original stitched dimensions: {originalWidth}x{originalHeight}")

        # Scale down if width exceeds 4K (3840 pixels)
        maxWidth = 3840  # 4K width
        if originalWidth > maxWidth:
            scaleFactor = maxWidth / originalWidth
            outputWidth = maxWidth
            outputHeight = int(originalHeight * scaleFactor)
        else:
            outputWidth, outputHeight = originalWidth, originalHeight

        # Ensure dimensions are even (required by H.264)
        if outputWidth % 2 != 0:
            outputWidth -= 1
        if outputHeight % 2 != 0:
            outputHeight -= 1

        print(f"Final output dimensions: {outputWidth}x{outputHeight}")

        # Reset video captures
        leftCap.release()
        middleCap.release()
        rightCap.release()
        leftCap = cv2.VideoCapture(self.leftVideo)
        middleCap = cv2.VideoCapture(self.middleVideo)
        rightCap = cv2.VideoCapture(self.rightVideo)

        totalFrames = min(
            int(leftCap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(middleCap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(rightCap.get(cv2.CAP_PROP_FRAME_COUNT))
        )

        # Set up FFmpeg process with proper dimensions
        command = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{outputWidth}x{outputHeight}',  # Use the new dimensions
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', 'fd:',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            outputPath
        ]

        # Start FFmpeg process
        process = Popen(command, stdin=PIPE)

        print(f"Processing and encoding {totalFrames} frames...")
        frame_count = 0

        try:
            while True:
                # Read frames from each video
                leftRet, leftFrame = leftCap.read()
                middleRet, middleFrame = middleCap.read()
                rightRet, rightFrame = rightCap.read()

                # Break if any video ends
                if not (leftRet and middleRet and rightRet):
                    break

                print(f"Processing frame {frame_count} of {totalFrames}")

                # Process frames with pre-computed overlap
                processedFrame = self._processFrame(
                    leftFrame, middleFrame, rightFrame,
                    transformationMatrices,
                    precomputedOverlap=precomputedOverlap
                )

                # Resize to the new dimensions
                processedFrame = cv2.resize(processedFrame, (outputWidth, outputHeight))

                # Write frame to FFmpeg process
                process.stdin.write(processedFrame.tobytes())

                frame_count += 1
                if frame_count >= totalFrames:
                    break

        finally:
            # Clean up resources
            leftCap.release()
            middleCap.release()
            rightCap.release()

            # Ensure FFmpeg process is properly closed
            if process.stdin:
                process.stdin.close()
            process.wait()

            print(f"Video successfully saved to {outputPath}")
            print(f"Processed and encoded {frame_count} frames")
