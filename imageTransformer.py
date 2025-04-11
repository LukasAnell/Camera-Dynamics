import cv2
from transformationMatrixMaker import transformationMatrixMaker

class ImageTransformer:
    def __init__(self, leftImage, middleImage, rightImage, cameraFocalHeight, cameraFocalLength, projectionPlaneDistanceFromCenter, imageDimensions):
        self.leftImage = leftImage
        self.middleImage = middleImage
        self.rightImage = rightImage
        self.cameraFocalHeight = cameraFocalHeight
        self.cameraFocalLength = cameraFocalLength
        self.projectionPlaneDistanceFromCenter = projectionPlaneDistanceFromCenter
        self.imageDimensions = imageDimensions  # (width, height)

    def getTransformationMatrix(self, cameraPosition, cameraForwardVector):
        """Generate a transformation matrix for the given camera parameters."""
        return transformationMatrixMaker(
            cameraPosition=cameraPosition,
            cameraForwardVector=cameraForwardVector,
            cameraFocalHeight=self.cameraFocalHeight,
            cameraFocalLength=self.cameraFocalLength,
            projectionPlaneDistanceFromCenter=self.projectionPlaneDistanceFromCenter,
            imageDimensions=self.imageDimensions
        )

    def applyTransformation(self, image, transformationMatrix):
        """Apply the transformation matrix to the given image."""
        return cv2.warpPerspective(image, transformationMatrix, self.imageDimensions)

    def transformLeftImage(self, cameraPosition, cameraForwardVector):
        """Transform the left image using the transformation matrix."""
        transformationMatrix = self.getTransformationMatrix(cameraPosition, cameraForwardVector)
        self.leftImage = self.applyTransformation(self.leftImage, transformationMatrix)

    def transformRightImage(self, cameraPosition, cameraForwardVector):
        """Transform the right image using the transformation matrix."""
        transformationMatrix = self.getTransformationMatrix(cameraPosition, cameraForwardVector)
        self.rightImage = self.applyTransformation(self.rightImage, transformationMatrix)

    def stitchImages(self):
        """Stitch the left, middle, and right images together."""
        return cv2.hconcat([self.leftImage, self.middleImage, self.rightImage])

    def showStitchedImage(self):
        """Display the stitched image."""
        stitchedImage = self.stitchImages()
        cv2.imshow('Stitched Image', stitchedImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()