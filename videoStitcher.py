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


    def outputStitchedVideo(self, fileName: str, outputDir: str = "Outputs"):
        import os
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        # Initialize video captures
        leftCap = cv2.VideoCapture(self.leftVideo)
        middleCap = cv2.VideoCapture(self.middleVideo)
        rightCap = cv2.VideoCapture(self.rightVideo)

        # Get frame rate from source video (instead of hardcoded 60)
        fps = leftCap.get(cv2.CAP_PROP_FPS)

        # Read first frames to determine actual output dimensions
        leftRet, leftFrame = leftCap.read()
        middleRet, middleFrame = middleCap.read()
        rightRet, rightFrame = rightCap.read()

        if not (leftRet and middleRet and rightRet):
            raise ValueError("Failed to read first frame from one or more videos")

        # Create transformer and get actual dimensions
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
        firstTransformer.transformLeftImage()
        firstTransformer.transformMiddleImage()
        firstTransformer.transformRightImage()

        # Reset video captures
        leftCap.release()
        middleCap.release()
        rightCap.release()
        leftCap = cv2.VideoCapture(self.leftVideo)
        middleCap = cv2.VideoCapture(self.middleVideo)
        rightCap = cv2.VideoCapture(self.rightVideo)

        # Get actual output dimensions from first stitched frame
        test_stitched = firstTransformer.stitchImages()
        outputHeight, output_width = test_stitched.shape[:2]

        # Try different codecs in order of preference, starting with ones that don't require external libraries
        codec_options = [
            ('MJPG', '.avi'),  # Motion JPEG in AVI container - widely compatible
            ('XVID', '.avi'),  # XVID codec in AVI container
            ('mp4v', '.mp4'),  # Basic MPEG-4 codec
            ('avc1', '.mp4')  # H.264 (requires OpenH264 library)
        ]

        # Try each codec until one works
        output = None
        used_codec = None
        file_ext = None

        for codec, ext in codec_options:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                outputPath = os.path.join(outputDir, fileName + ext)

                # Get dimensions from test_stitched (using your existing code)
                test_stitched = firstTransformer.stitchImages()
                outputHeight, outputWidth = test_stitched.shape[:2]

                # Try to create VideoWriter with this codec
                output = cv2.VideoWriter(outputPath, fourcc, fps, (outputWidth, outputHeight))

                if output.isOpened():
                    used_codec = codec
                    file_ext = ext
                    break
                else:
                    output.release()  # Clean up if not opened successfully
            except Exception as e:
                print(f"Failed to use codec {codec}: {str(e)}")

        if output is None or not output.isOpened():
            raise Exception("Failed to initialize any video codec. Try installing FFmpeg or OpenH264.")

        print(f"Using codec: {used_codec} with container: {file_ext}")

        if not output.isOpened():
            raise Exception(f"Failed to create output video file at {outputPath}")

        totalFrames = min(
            int(leftCap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(middleCap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(rightCap.get(cv2.CAP_PROP_FRAME_COUNT))
        )

        frameNumber = 0
        try:
            while frameNumber < totalFrames:
                leftRet, leftFrame = leftCap.read()
                middleRet, middleFrame = middleCap.read()
                rightRet, rightFrame = rightCap.read()

                if not (leftRet and middleRet and rightRet):
                    print(f"End of video reached or error reading frame {frameNumber}. Stopping.")
                    break

                print(f"Processing frame {frameNumber} of {totalFrames}")
                # if frameNumber % 10 == 0:
                #     print(f"Processing frame {frameNumber} of {totalFrames}")

                if leftFrame is None or middleFrame is None or rightFrame is None:
                    print(f"One of the frames is None at frame {frameNumber}. Skipping this frame.")
                    continue

                ImageTransformer = imageTransformer.ImageTransformer(
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

                ImageTransformer.transformLeftImage()
                ImageTransformer.transformMiddleImage()
                ImageTransformer.transformRightImage()

                # Write the stitched image to the output video
                # if not output.isOpened():
                #     print(f"Warning: Output video writer not opened at frame {frameNumber}. Trying to reopen.")
                #     output = cv2.VideoWriter(outputPath, fourcc, fps, (outputWidth, outputHeight))
                #     if not output.isOpened():
                #         raise Exception(f"Failed to reopen output video writer at frame {frameNumber}")
                stitchedFrame = ImageTransformer.stitchImages()
                output.write(stitchedFrame)
                frameNumber += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            print(f"Finished processing {frameNumber} frames.")
            leftCap.release()
            middleCap.release()
            rightCap.release()
            output.release()
            cv2.destroyAllWindows()
        return
