import math

import cv2
import numpy as np
from transformationMatrixMaker import transformationMatrixMaker

class ImageTransformer:
    def __init__(self, leftImage, middleImage, rightImage, leftAngle, rightAngle, cameraFocalHeight, cameraFocalLength, projectionPlaneDistanceFromCenter, imageDimensions):
        self.leftImage = leftImage
        self.middleImage = middleImage
        self.rightImage = rightImage
        self.leftAngle = leftAngle
        self.rightAngle = rightAngle
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


    def getLeftForwardVectorAndPosition(self):
        radians = math.radians(self.leftAngle)
        cameraPosition = (math.cos(radians), math.sin(radians), 0)
        cameraForwardVector = (math.cos(radians), math.sin(radians), 0)
        return cameraPosition, cameraForwardVector


    def getRightForwardVectorAndPosition(self):
        radians = math.radians(self.rightAngle)
        cameraPosition = (math.cos(radians), math.sin(radians), 0)
        cameraForwardVector = (math.cos(radians), math.sin(radians), 0)
        return cameraPosition, cameraForwardVector


    def applyTransformation(self, image, transformationMatrix):
        """Apply the transformation matrix to the given image."""
        return cv2.warpPerspective(image, transformationMatrix, self.imageDimensions)


    def transformLeftImage(self):
        cameraPosition, cameraForwardVector = self.getLeftForwardVectorAndPosition()
        """Transform the left image using the transformation matrix."""
        transformationMatrix = self.getTransformationMatrix(cameraPosition, cameraForwardVector)
        self.leftImage = self.applyTransformation(cv2.flip(self.leftImage, 1), transformationMatrix)


    def transformMiddleImage(self):
        cameraPosition = (0, 0, 0)
        cameraForwardVector = (1, 0, 0)
        """Transform the middle image using the transformation matrix."""
        transformationMatrix = self.getTransformationMatrix(cameraPosition, cameraForwardVector)
        self.middleImage = self.applyTransformation(cv2.flip(self.middleImage, 1), transformationMatrix)


    def transformRightImage(self):
        cameraPosition, cameraForwardVector = self.getRightForwardVectorAndPosition()
        """Transform the right image using the transformation matrix."""
        transformationMatrix = self.getTransformationMatrix(cameraPosition, cameraForwardVector)
        self.rightImage = self.applyTransformation(cv2.flip(self.rightImage,1), transformationMatrix)


    def stitchImages(self):
        """Stitch the left, middle, and right images together."""
        # Get the original image height from imageDimensions
        original_height = self.imageDimensions[1]

        # Get the dimensions of the transformed images
        left_height, left_width = self.leftImage.shape[:2]
        middle_height, middle_width = self.middleImage.shape[:2]
        right_height, right_width = self.rightImage.shape[:2]

        # Calculate the scale factor to resize the middle image to match the original height
        # while preserving its aspect ratio
        scale_factor = original_height / middle_height
        middle_resized_width = int(middle_width * scale_factor)
        middle_resized = cv2.resize(self.middleImage, (middle_resized_width, original_height))

        # Create a blank canvas for the middle image with empty space above and below
        # The height will be the same as the original image
        middle_canvas = np.zeros((original_height, middle_resized_width, 3), dtype=np.uint8)

        # Calculate the vertical position to place the middle image in the center
        # If we want to keep the original middle image height and add padding
        # We'll resize it to a smaller height and place it in the center
        target_height = middle_height  # Keep the original middle image height
        scale_factor_height = target_height / original_height
        middle_small_width = int(middle_resized_width * scale_factor_height)
        middle_small_height = target_height

        # Resize the middle image to the target height while preserving aspect ratio
        middle_small = cv2.resize(self.middleImage, (middle_small_width, middle_small_height))

        # Calculate the position to place the small image in the center of the canvas
        y_offset = (original_height - middle_small_height) // 2
        x_offset = (middle_resized_width - middle_small_width) // 2

        # Place the small middle image in the center of the canvas
        middle_canvas[y_offset:y_offset+middle_small_height, x_offset:x_offset+middle_small_width] = middle_small

        # Resize left and right images to match the height of the original image
        left_resized = cv2.resize(self.leftImage, (int(left_width * (original_height / left_height)), original_height))
        right_resized = cv2.resize(self.rightImage, (int(right_width * (original_height / right_height)), original_height))

        # Concatenate the resized images
        return cv2.hconcat([left_resized, middle_canvas, right_resized])


    def showStitchedImage(self):
        """Display the stitched image."""
        stitchedImage = self.stitchImages()
        cv2.imshow('Stitched Image', stitchedImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def saveStitchedImage(self, filename):
        """Save the stitched image to a file."""
        stitchedImage = self.stitchImages()
        cv2.imwrite(filename, stitchedImage)
