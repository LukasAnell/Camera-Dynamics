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
        try:
            fourcc = cv2.VideoWriter_fourcc(*'H264')  # Try H264 codec first
        except:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        # Ensure output directory exists
        import os
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        output = cv2.VideoWriter(os.path.join(outputDir, fileName + ".mp4"), fourcc, 60.0, (self.imageDimensions[0] * 3, self.imageDimensions[1]))

        leftCap = cv2.VideoCapture(self.leftVideo)
        middleCap = cv2.VideoCapture(self.middleVideo)
        rightCap = cv2.VideoCapture(self.rightVideo)

        totalFrames = min(
            int(leftCap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(middleCap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(rightCap.get(cv2.CAP_PROP_FRAME_COUNT))
        )

        # Initialize with first frames
        leftRet, leftFrame = leftCap.read()
        middleRet, middleFrame = middleCap.read()
        rightRet, rightFrame = rightCap.read()

        # Validate first frames
        if not (leftRet and middleRet and rightRet):
            raise ValueError("Failed to read first frame from one or more videos")

        ImageTransformer = imageTransformer.ImageTransformer(
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
        transformationMatrices = ImageTransformer.initializeTransformationMatrices()

        frameNumber = 0
        while frameNumber < totalFrames:
            leftRet, leftFrame = leftCap.read()
            middleRet, middleFrame = middleCap.read()
            rightRet, rightFrame = rightCap.read()

            if not (leftRet and middleRet and rightRet):
                print(f"End of video reached or error reading frame {frameNumber}. Stopping.")
                break

            if frameNumber % 50 == 0:
                print(f"Processing frame {frameNumber} of {totalFrames}")

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
            output.write(ImageTransformer.stitchImages())

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frameNumber += 1

        leftCap.release()
        middleCap.release()
        rightCap.release()
        output.release()
        cv2.destroyAllWindows()
        return
