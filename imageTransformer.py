import math

import cv2
import numpy as np

class ImageTransformer:
    def __init__(self,
            leftImage: np.ndarray,
            middleImage: np.ndarray,
            rightImage: np.ndarray,
            leftAngle: int,
            rightAngle: int,
            cameraFocalHeight: float,
            cameraFocalLength: float,
            projectionPlaneDistanceFromCenter: float,
            imageDimensions: (int, int),
            transformationMatrices: list = []
    ):
        self.leftImage = leftImage
        self.middleImage = middleImage
        self.rightImage = rightImage
        self.leftAngle = leftAngle
        self.rightAngle = rightAngle
        self.cameraFocalHeight = cameraFocalHeight
        self.cameraFocalLength = cameraFocalLength
        self.projectionPlaneDistanceFromCenter = projectionPlaneDistanceFromCenter
        self.imageDimensions = imageDimensions  # (width, height)
        self.transformationMatrices = transformationMatrices


    def setTransformationMatrices(self, transformationMatrices):
        """Set the transformation matrices for the left, middle, and right images."""
        self.transformationMatrices = transformationMatrices


    # Initialize transformation matrices for left, middle, and right images
    def initializeTransformationMatrices(self):
        cameraPosition, cameraForwardVector = self.getLeftForwardVectorAndPosition()
        leftTransformationMatrix = self.getTransformationMatrix(
            cameraPosition=cameraPosition,
            cameraForwardVector=cameraForwardVector
        )
        self.transformationMatrices.append(leftTransformationMatrix)
        cameraPosition, cameraForwardVector = self.getMiddleForwardVectorAndPosition()
        middleTransformationMatrix = self.getTransformationMatrix(
            cameraPosition=cameraPosition,
            cameraForwardVector=cameraForwardVector
        )
        self.transformationMatrices.append(middleTransformationMatrix)
        cameraPosition, cameraForwardVector = self.getRightForwardVectorAndPosition()
        rightTransformationMatrix = self.getTransformationMatrix(
            cameraPosition=cameraPosition,
            cameraForwardVector=cameraForwardVector
        )
        self.transformationMatrices.append(rightTransformationMatrix)
        return self.transformationMatrices


    def getTransformationMatrix(self,
            cameraPosition,
            cameraForwardVector
    ):
        """Generate a transformation matrix for the given camera parameters."""
        return self.transformationMatrixMaker(
            cameraPosition=cameraPosition,
            cameraForwardVector=cameraForwardVector,
            cameraFocalHeight=self.cameraFocalHeight,
            cameraFocalLength=self.cameraFocalLength,
            projectionPlaneDistanceFromCenter=self.projectionPlaneDistanceFromCenter,
            imageDimensions=self.imageDimensions
        )[0]


    def getStartingEndingCoordinates(self,
            cameraPosition,
            cameraForwardVector
    ):
        # Can be obtained by using transformationMatrixMaker and storing the 2nd and 3rd elements of the tuple
        transformationMatrix = self.transformationMatrixMaker(
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

    def getScalingFactor(self,
            cameraPosition,
            cameraForwardVector
    ):
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


    def applyTransformation(self,
            image,
            transformationMatrix
    ):
        """Apply the transformation matrix to the given image."""
        return cv2.warpPerspective(image, transformationMatrix, self.imageDimensions)


    def transformLeftImage(self):
        cameraPosition, cameraForwardVector = self.getLeftForwardVectorAndPosition()
        """Transform the left image using the transformation matrix."""
        transformationMatrix = self.transformationMatrices[0]
        self.leftImage = self.applyTransformation(cv2.flip(self.leftImage, 1), transformationMatrix)


    def transformMiddleImage(self):
        cameraPosition = (0, 0, 0)
        cameraForwardVector = (1, 0, 0)
        """Transform the middle image using the transformation matrix."""
        transformationMatrix = self.transformationMatrices[1]
        self.middleImage = self.applyTransformation(cv2.flip(self.middleImage, 1), transformationMatrix)


    def transformRightImage(self):
        cameraPosition, cameraForwardVector = self.getRightForwardVectorAndPosition()
        """Transform the right image using the transformation matrix."""
        transformationMatrix = self.transformationMatrices[2]
        self.rightImage = self.applyTransformation(cv2.flip(self.rightImage,1), transformationMatrix)

    def stitchImages(self):
        """
        Stitch the left, middle, and right images together.
        This function:
        1. Calculates scaling factors for left and right images
        2. Scales up the left and right images to match the correct height
        3. Adds black bars to the middle image to match the height of the scaled side images
        4. Concatenates all three images horizontally
        """
        # Get scaling factors for left and right images
        leftPos, leftFwd = self.getLeftForwardVectorAndPosition()
        rightPos, rightFwd = self.getRightForwardVectorAndPosition()

        leftScalingFactor = self.getScalingFactor(leftPos, leftFwd)
        rightScalingFactor = self.getScalingFactor(rightPos, rightFwd)

        # Get current dimensions of all images
        leftHeight, leftWidth = self.leftImage.shape[:2]
        middleHeight, middleWidth = self.middleImage.shape[:2]
        rightHeight, rightWidth = self.rightImage.shape[:2]

        # Scale up the left and right images
        scaledLeftHeight = int(leftHeight * leftScalingFactor)
        scaledRightHeight = int(rightHeight * rightScalingFactor)

        # Determine the maximum height needed for all images
        maxHeight = max(scaledLeftHeight, scaledRightHeight)

        # Resize left and right images to maintain the aspect ratio while scaling to the correct height
        scaledLeftWidth = int(leftWidth * (scaledLeftHeight / leftHeight))
        scaledRightWidth = int(rightWidth * (scaledRightHeight / rightHeight))

        scaledLeftImage = cv2.resize(self.leftImage, (scaledLeftWidth, scaledLeftHeight))
        scaledRightImage = cv2.resize(self.rightImage, (scaledRightWidth, scaledRightHeight))

        # Add black bars to the middle image to match the height of the scaled side images
        # Calculate the padding needed at the top and bottom
        paddingTop = (maxHeight - middleHeight) // 2
        paddingBottom = maxHeight - middleHeight - paddingTop

        # Create a black image with the desired height and same width as middle image
        paddedMiddleImage = np.zeros((maxHeight, middleWidth, 3), dtype=np.uint8)

        # Place the middle image in the center of the padded image
        paddedMiddleImage[paddingTop:paddingTop + middleHeight, :] = self.middleImage

        # Stitch all three images together horizontally
        stitchedImage = cv2.hconcat([scaledLeftImage, paddedMiddleImage, scaledRightImage])

        return stitchedImage


    def showStitchedImage(self):
        """Display the stitched image."""
        stitchedImage = self.stitchImages()
        cv2.imshow('Stitched Image', stitchedImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def saveStitchedImage(self,
            filename
    ):
        """Save the stitched image to a file."""
        stitchedImage = self.stitchImages()
        cv2.imwrite(filename, stitchedImage)


    def removeOverlap(self):
        # Get camera positions and forward vectors for all three cameras
        leftPos, leftFwd = self.getLeftForwardVectorAndPosition()
        middlePos, middleFwd = self.getMiddleForwardVectorAndPosition()
        rightPos, rightFwd = self.getRightForwardVectorAndPosition()

        # Get pre-transformation coordinates for all three images
        _, _, _, leftPreTransCoords = self.transformationMatrixMaker(
            leftPos, leftFwd, self.cameraFocalHeight, self.cameraFocalLength,
            self.projectionPlaneDistanceFromCenter, self.imageDimensions
        )

        _, _, _, middlePreTransCoords = self.transformationMatrixMaker(
            middlePos, middleFwd, self.cameraFocalHeight, self.cameraFocalLength,
            self.projectionPlaneDistanceFromCenter, self.imageDimensions
        )

        _, _, _, rightPreTransCoords = self.transformationMatrixMaker(
            rightPos, rightFwd, self.cameraFocalHeight, self.cameraFocalLength,
            self.projectionPlaneDistanceFromCenter, self.imageDimensions
        )

        leftXMin = min(leftPreTransCoords[0][0], leftPreTransCoords[1][0])
        leftXMax = max(leftPreTransCoords[2][0], leftPreTransCoords[3][0])
        middleXMin = min(middlePreTransCoords[0][0], middlePreTransCoords[1][0])
        middleXMax = max(middlePreTransCoords[2][0], middlePreTransCoords[3][0])
        rightXMin = min(rightPreTransCoords[0][0], rightPreTransCoords[1][0])
        rightXMax = max(rightPreTransCoords[2][0], rightPreTransCoords[3][0])

        # Find the overlap area between the left and middle images
        leftMiddleOverlapXMin = max(leftXMin, middleXMin)
        leftMiddleOverlapXMax = min(leftXMax, middleXMax)
        leftMiddleOverlapWidth = max(0, leftMiddleOverlapXMax - leftMiddleOverlapXMin)

        # Find the overlap area between the middle and right images
        rightMiddleOverlapXMin = max(rightXMin, middleXMin)
        rightMiddleOverlapXMax = min(rightXMax, middleXMax)
        rightMiddleOverlapWidth = max(0, rightMiddleOverlapXMax - rightMiddleOverlapXMin)

        # Find the scaling factors and scale up the overlap areas
        leftMiddleScalingFactor = self.getScalingFactor(leftPos, leftFwd)
        rightMiddleScalingFactor = self.getScalingFactor(rightPos, rightFwd)



    def transformationMatrixMaker (self,
            cameraPosition: [],
            cameraForwardVector: [],
            cameraFocalHeight: float,
            cameraFocalLength: float,
            projectionPlaneDistanceFromCenter: float,
            imageDimensions: (int, int)  # width, height
    ):
        # first construct camera bounding corners
        focalAngle: float = math.atan(cameraFocalHeight / cameraFocalLength)
        # because we're initially facing in the (1, 0, 0) direction
        # we already know the vectors on the top and left bounding planes are going
        # to be expressed as
        # Camera top plane vector = [cos(focalAngle), 0, sin(focalAngle)]
        # Camera left plane vector = [cos(focalAngle), sin(focalAngle), 0]
        cameraPositiveZPlaneVector = [math.cos(focalAngle), 0, math.sin(focalAngle)]  # Fy
        cameraPositiveYPlaneVector = [math.cos(focalAngle), math.sin(focalAngle), 0]  # Fx
        # going counterclockwise starting in quadrant 1 from the view of the camera
        topRightCornerVector = np.array(cameraPositiveYPlaneVector) - 2 * (
                    np.array(cameraPositiveYPlaneVector) - np.array([math.sqrt(2) / 2, 0, 0])) + (
                                               np.array(cameraPositiveZPlaneVector) - np.array(
                                           [math.sqrt(2) / 2, 0, 0]
                                       ))
        bottomLeftCornerVector = np.array(cameraPositiveYPlaneVector) - (
                    np.array(cameraPositiveZPlaneVector) - np.array([math.sqrt(2) / 2, 0, 0]))

        bottomRightCornerVector = np.copy(topRightCornerVector)
        bottomRightCornerVector[2] = -topRightCornerVector[2]
        topLeftCornerVector = np.copy(bottomLeftCornerVector)
        topLeftCornerVector[2] = -bottomLeftCornerVector[2]

        finalABPositions = [[0, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]]

        # compute the angle between cameras
        # a dot b = |a| * |b| cos(theta)
        # we are assuming magnitude of a and b are 1
        angleBetweenCameras = math.acos(np.dot(cameraForwardVector, [1, 0, 0]))
        if cameraForwardVector[1] < 0:
            angleBetweenCameras = -angleBetweenCameras

        # construct rotation matrix
        """
        cosθ, -sinθ, 0
        sinθ, cosθ , 0
        0   , 0    , 1
        """
        zRotationMatrix = [
            [math.cos(angleBetweenCameras), -math.sin(angleBetweenCameras), 0],
            [math.sin(angleBetweenCameras), math.cos(angleBetweenCameras), 0],
            [0, 0, 1]
        ]
        topRightCornerVector = np.matmul(zRotationMatrix, topRightCornerVector)
        bottomRightCornerVector = np.matmul(zRotationMatrix, bottomRightCornerVector)
        bottomLeftCornerVector = np.matmul(zRotationMatrix, bottomLeftCornerVector)
        topLeftCornerVector = np.matmul(zRotationMatrix, topLeftCornerVector)

        # find intersection with delta plane
        # lineParameter = (projectionPlaneDistanceFromCenter - cameraPosition[0]) / (cameraForwardVector[0])
        # sticking everything in an array for convenience
        projectedLines = [cameraForwardVector, topRightCornerVector, topLeftCornerVector, bottomLeftCornerVector,
                          bottomRightCornerVector]
        finalCDPositions = []
        for projectedLineVector in projectedLines:
            lineParameter = (projectionPlaneDistanceFromCenter - cameraPosition[0]) / projectedLineVector[0]
            c = cameraPosition[1] + lineParameter * projectedLineVector[1]
            d = cameraPosition[2] + lineParameter * projectedLineVector[2]
            finalCDPositions += [[c, d]]
        preTransformationCDPositions = finalCDPositions

        # Find the minimum x and y values for finalABPositions
        minXAB = min(pos[0] for pos in finalABPositions)
        minYAB = min(pos[1] for pos in finalABPositions)

        # Shift all coordinates in finalABPositions to be within the first quadrant
        for i in range(len(finalABPositions)):
            finalABPositions[i][0] -= minXAB
            finalABPositions[i][1] -= minYAB

        # Find the minimum x and y values for finalCDPositions
        minXCD = min(pos[0] for pos in finalCDPositions)
        minYCD = min(pos[1] for pos in finalCDPositions)

        # Shift all coordinates in finalCDPositions to be within the first quadrant
        for i in range(len(finalCDPositions)):
            finalCDPositions[i][0] -= minXCD
            finalCDPositions[i][1] -= minYCD

        # Calculate the scaling factors
        imageWidth, imageHeight = imageDimensions
        maxXAB = max(pos[0] for pos in finalABPositions)
        maxYAB = max(pos[1] for pos in finalABPositions)
        xScalingFactorAB = imageWidth / maxXAB
        yScalingFactorAB = imageHeight / maxYAB

        maxXCD = max(pos[0] for pos in finalCDPositions)
        maxYCD = max(pos[1] for pos in finalCDPositions)
        xScalingFactorCD = imageWidth / maxXCD
        yScalingFactorCD = imageHeight / maxYCD

        # Apply the scaling factors to each coordinate in finalABPositions
        for i in range(len(finalABPositions)):
            finalABPositions[i][0] *= xScalingFactorAB
            finalABPositions[i][1] *= yScalingFactorAB

        # Apply the scaling factors to each coordinate in finalCDPositions
        for i in range(len(finalCDPositions)):
            finalCDPositions[i][0] *= xScalingFactorCD
            finalCDPositions[i][1] *= yScalingFactorCD

        finalABPositions = [
            finalABPositions[3],  # top-left
            finalABPositions[1],  # top-right
            finalABPositions[4],  # bottom-right
            finalABPositions[2]  # bottom-left
        ]

        finalCDPositions = [
            finalCDPositions[3],  # top-left
            finalCDPositions[1],  # top-right
            finalCDPositions[4],  # bottom-right
            finalCDPositions[2]  # bottom-left
        ]

        # use cv2.getPerspectiveTransform to get transformation matrix
        # AB is the source, CD is the destination
        # Ensure AB and CD are numpy arrays of type float32 and contain exactly 4 points
        AB = np.array(finalABPositions, dtype=np.float32)
        CD = np.array(finalCDPositions, dtype=np.float32)

        transformationMatrix = cv2.getPerspectiveTransform(AB, CD)
        # print("AB:\n", AB)
        # print("CD:\n", CD)
        return transformationMatrix, AB, CD, preTransformationCDPositions
