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
        )[0]


    def getStartingEndingCoordinates(self, cameraPosition, cameraForwardVector):
        # Can be obtained by using transformationMatrixMaker and storing the 2nd and 3rd elements of the tuple
        transformationMatrix = transformationMatrixMaker(
            cameraPosition=cameraPosition,
            cameraForwardVector=cameraForwardVector,
            cameraFocalHeight=self.cameraFocalHeight,
            cameraFocalLength=self.cameraFocalLength,
            projectionPlaneDistanceFromCenter=self.projectionPlaneDistanceFromCenter,
            imageDimensions=self.imageDimensions
        )
        startingCoordinates = transformationMatrix[1]
        endingCoordinates = transformationMatrix[2]
        return startingCoordinates, endingCoordinates

    def getScalingFactor (self, cameraPosition, cameraForwardVector):
        # Using the starting and ending coordinates of the images, find the pinched edge and use that to find the scaling factor between the original image and the transformed image
        startingCoordinates, endingCoordinates = self.getStartingEndingCoordinates(cameraPosition, cameraForwardVector)
        # starting and ending coordinates are in the form of [(x,y), (x,y), (x,y), (x,y), (x,y)]
        # The pinched edge is the smallest distance between two vertically aligned points in the ending coordinates array, which may vary depending on the angle of the camera
        # Calculate the height of the pinched edge by finding the distance between the two points
        # Only need to compare the y coordinates of the points, and only compare two points whose x coordinates are the same
        pinchedEdgeHeight = float('inf')  # Initialize to infinity
        for i in range(len(endingCoordinates)):
            for j in range(i + 1, len(endingCoordinates)):
                if endingCoordinates[i][0] == endingCoordinates[j][0]:
                    pinchedEdgeHeight = min(pinchedEdgeHeight, abs(endingCoordinates[i][1] - endingCoordinates[j][1]))
        # The scaling factor is the ratio of the height of the original image to the height of the pinched edge
        # The height of the original image is the difference between the maximum and minimum y coordinates of the starting coordinates
        originalImageHeight = max([coord[1] for coord in startingCoordinates]) - min(
            [coord[1] for coord in startingCoordinates]
        )
        # The scaling factor is the ratio of the original image height to the pinched-edge height
        scalingFactor = originalImageHeight / pinchedEdgeHeight
        return scalingFactor


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


    def getMiddleForwardVectorAndPosition(self):
        cameraPosition = (0, 0, 0)
        cameraForwardVector = (1, 0, 0)
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

    def stitchImages (self):
        """
        Stitch the left, middle, and right images together.
        This function:
        1. Calculates scaling factors for left and right images
        2. Scales up the left and right images to match the correct height
        3. Adds black bars to the middle image to match the height of the scaled side images
        4. Concatenates all three images horizontally
        """
        # Get scaling factors for left and right images
        left_pos, left_fwd = self.getLeftForwardVectorAndPosition()
        right_pos, right_fwd = self.getRightForwardVectorAndPosition()

        left_scaling_factor = self.getScalingFactor(left_pos, left_fwd)
        right_scaling_factor = self.getScalingFactor(right_pos, right_fwd)

        # Get current dimensions of all images
        left_height, left_width = self.leftImage.shape[:2]
        middle_height, middle_width = self.middleImage.shape[:2]
        right_height, right_width = self.rightImage.shape[:2]

        # Scale up the left and right images
        scaled_left_height = int(left_height * left_scaling_factor)
        scaled_right_height = int(right_height * right_scaling_factor)

        # Determine the maximum height needed for all images
        max_height = max(scaled_left_height, scaled_right_height)

        # Resize left and right images to maintain aspect ratio while scaling to the correct height
        scaled_left_width = int(left_width * (scaled_left_height / left_height))
        scaled_right_width = int(right_width * (scaled_right_height / right_height))

        scaled_left_image = cv2.resize(self.leftImage, (scaled_left_width, scaled_left_height))
        scaled_right_image = cv2.resize(self.rightImage, (scaled_right_width, scaled_right_height))

        # Add black bars to the middle image to match the height of the scaled side images
        # Calculate the padding needed at the top and bottom
        padding_top = (max_height - middle_height) // 2
        padding_bottom = max_height - middle_height - padding_top

        # Create a black image with the desired height and same width as middle image
        padded_middle_image = np.zeros((max_height, middle_width, 3), dtype=np.uint8)

        # Place the middle image in the center of the padded image
        padded_middle_image[padding_top:padding_top + middle_height, :] = self.middleImage

        # Stitch all three images together horizontally
        stitched_image = cv2.hconcat([scaled_left_image, padded_middle_image, scaled_right_image])

        return stitched_image


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
