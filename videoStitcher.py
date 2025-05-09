import cv2

import imageTransformer
from imageTransformer import ImageTransformer


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


    def outputStitchedVideo(self, fileName: str):
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        output = cv2.VideoWriter("Outputs/" + fileName + ".mp4", fourcc, 60.0, (self.imageDimensions[0] * 3, self.imageDimensions[1]))

        leftCap = cv2.VideoCapture(self.leftVideo)
        middleCap = cv2.VideoCapture(self.middleVideo)
        rightCap = cv2.VideoCapture(self.rightVideo)

        frameNumber = -1
        leftRet, leftFrame = leftCap.read()
        middleRet, middleFrame = middleCap.read()
        rightRet, rightFrame = rightCap.read()

        while leftRet and middleRet and rightRet and frameNumber <= middleCap.get(cv2.CAP_PROP_FRAME_COUNT):
            frameNumber += 1
            leftRet, leftFrame = leftCap.read()
            middleRet, middleFrame = middleCap.read()
            rightRet, rightFrame = rightCap.read()

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

            ImageTransformer.transformLeftImage()
            ImageTransformer.transformMiddleImage()
            ImageTransformer.transformRightImage()
            output.write(ImageTransformer.stitchImages())

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        leftCap.release()
        middleCap.release()
        rightCap.release()
        output.release()
        cv2.destroyAllWindows()
        return