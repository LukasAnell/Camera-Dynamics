import cv2


class imageManipulation:
    def __init__(self, images: [], cameraFocalHeight, cameraFocalLength, projectionPlaneDistanceFromCenter, imageDimensions):
        self.leftImage = images[0]
        self.middleImage = images[1]
        self.rightImage = images[2]

        self.cameraFocalHeight = cameraFocalHeight
        self.cameraFocalLength = cameraFocalLength
        self.projectionPlaneDistanceFromCenter = projectionPlaneDistanceFromCenter
        self.imageDimensions = imageDimensions # (width, height)


    def getAllImages(self):
        return [self.leftImage, self.middleImage, self.rightImage]


    def getLeftImage(self):
        return self.leftImage


    def getMiddleImage(self):
        return self.middleImage


    def getRightImage(self):
        return self.rightImage


    def setLeftImage(self, leftImage):
        self.leftImage = leftImage


    def setMiddleImage(self, middleImage):
        self.middleImage = middleImage


    def setRightImage(self, rightImage):
        self.rightImage = rightImage


    def appendImages(self):
        # Use cv2.hconcat to concatenate the images horizontally
        stitchedImage = cv2.hconcat([self.leftImage, self.middleImage, self.rightImage])
        return stitchedImage


    def showStitchedImage(self):
        stitchedImage = self.appendImages()
        cv2.imshow('Stitched Image', stitchedImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Use transformationMatrixMaker() in transformationMatrixMaker.py to create a transformation matrix for one of the three images based on camera parameters
    """
    Parameters:
    cameraPosition: [],
    cameraForwardVector: [],
    cameraFocalHeight: float,
    cameraFocalLength: float,
    projectionPlaneDistanceFromCenter: float,
    imageDimensions: (int, int) # width, height
    """
    def getTransformationMatrix(self, cameraPosition, cameraForwardVector):
        # Call the transformationMatrixMaker function from transformationMatrixMaker.py
        from transformationMatrixMaker import transformationMatrixMaker
        transformationMatrix = transformationMatrixMaker(
            cameraPosition=cameraPosition,
            cameraForwardVector=cameraForwardVector,
            cameraFocalHeight=self.cameraFocalHeight,
            cameraFocalLength=self.cameraFocalLength,
            projectionPlaneDistanceFromCenter=self.projectionPlaneDistanceFromCenter,
            imageDimensions=self.imageDimensions
        )
        return transformationMatrix


    def transformImage(self, image, transformationMatrix):
        # Apply the transformation matrix to the image
        transformedImage = cv2.warpPerspective(image, transformationMatrix, self.imageDimensions)
        return transformedImage


