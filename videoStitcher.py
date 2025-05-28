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


    def _processFrame (self, leftFrame, middleFrame, rightFrame, transformationMatrices, frameNumber=None, fps=None):
        """
        Process a single set of frames from the three cameras.

        Args:
            leftFrame: Frame from the left camera
            middleFrame: Frame from the middle camera
            rightFrame: Frame from the right camera
            transformationMatrices: Pre-computed transformation matrices
            frameNumber: Optional frame number for timestamp overlay
            fps: Optional frames per second for timestamp calculation

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

        # Stitch the transformed images
        stitchedFrame = transformer.stitchImages()

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

    def outputStitchedVideo (self, fileName: str, outputDir: str = "Outputs"):
        import os
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        outputPath = os.path.join(outputDir, f"{fileName}")

        # Initialize video captures
        leftCap = cv2.VideoCapture(self.leftVideo)
        middleCap = cv2.VideoCapture(self.middleVideo)
        rightCap = cv2.VideoCapture(self.rightVideo)

        # Get frame rate from source video
        fps = self.getCommonFrameRate()
        print(f"Source video frame rate: {fps} fps")

        # Read first frames to determine actual output dimensions
        leftRet, leftFrame = leftCap.read()
        middleRet, middleFrame = middleCap.read()
        rightRet, rightFrame = rightCap.read()

        if not (leftRet and middleRet and rightRet):
            raise ValueError("Failed to read first frame from one or more videos")

        # Create transformer and initialize transformation matrices
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

        # Process first frame to get dimensions
        firstFrame = self._processFrame(leftFrame, middleFrame, rightFrame, transformationMatrices)
        outputHeight, outputWidth = firstFrame.shape[:2]

        # Reset video captures
        leftCap.release()
        middleCap.release()
        rightCap.release()
        leftCap = cv2.VideoCapture(self.leftVideo)
        middleCap = cv2.VideoCapture(self.middleVideo)
        rightCap = cv2.VideoCapture(self.rightVideo)

        # Try different codecs in order of preference
        codec_options = [
            ('MJPG', '.avi'),  # Motion JPEG
            ('XVID', '.avi'),  # XVID codec in AVI container b
            ('mp4v', '.mp4'),  # MP4 with H.264 codec
            ('avc1', '.mp4'),  # Another variation of H.264
            ('WMV2', '.wmv')  # Windows Media Video
        ]

        # Try each codec until one works
        output = None
        used_codec = None
        file_ext = None

        for codec, ext in codec_options:
            file_path = outputPath + ext
            fourcc = cv2.VideoWriter_fourcc(*codec)
            temp_output = cv2.VideoWriter(file_path, fourcc, fps, (outputWidth, outputHeight))

            if temp_output.isOpened():
                output = temp_output
                used_codec = codec
                file_ext = ext
                break

        if output is None or not output.isOpened():
            raise Exception(f"Could not open video writer with any of the available codecs.")

        print(f"Using codec: {used_codec} with container: {file_ext}")

        if not output.isOpened():
            raise Exception(f"Failed to create output video file")

        totalFrames = min(
            int(leftCap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(middleCap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(rightCap.get(cv2.CAP_PROP_FRAME_COUNT))
        )

        # PHASE 1: Process all frames first and store them
        print("Phase 1: Processing all frames...")
        processedFrames = []
        for frameNumber in range(totalFrames):
            print(f"Processing frame {frameNumber} of {totalFrames}")

            leftRet, leftFrame = leftCap.read()
            middleRet, middleFrame = middleCap.read()
            rightRet, rightFrame = rightCap.read()

            if not (leftRet and middleRet and rightRet):
                print(f"End of video reached at frame {frameNumber}")
                break

            if leftFrame is None or middleFrame is None or rightFrame is None:
                print(f"One of the frames is None at frame {frameNumber}. Skipping.")
                continue

            # Process the frame using our helper method
            stitchedFrame = self._processFrame(
                leftFrame, middleFrame, rightFrame,
                transformationMatrices, frameNumber, fps
            )

            # Store the processed frame
            processedFrames.append(stitchedFrame)

        # PHASE 2: Write all frames at the correct FPS
        print(f"Phase 2: Writing {len(processedFrames)} frames to output video...")
        for i, frame in enumerate(processedFrames):
            # Write the frame
            if frame is None:
                print(f"Frame {i} is None, skipping write.")
                continue
            output.write(frame)

        # Clean up
        leftCap.release()
        middleCap.release()
        rightCap.release()
        output.release()
        print(f"Video successfully saved to {outputPath + file_ext}")
