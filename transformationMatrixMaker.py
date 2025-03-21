import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import math

# assume center at (0, 0, 0)
# assume camera initially faces (1, 0, 0)
# and we will rotate the camera to face that once we
# make the corner vectors

def main():
    # testMatrix = cv2.getPerspectiveTransform(
    #     [(0, 0), (300, 0), (300, 300), (0, 300)],
    #     [(0, 50), (300, 0), (300, 300), (0, 250)]
    # )
    # print(np.matrix(testMatrix))
    # sourceImage = cv2.imread("test2.jpg")
    # # convert transformationMatrix into a 2x3 matrix
    # transformationMatrix = np.float32(np.reshape(testMatrix[:2], (2, 3)))
    # imageTransformed = cv2.warpAffine(sourceImage, transformationMatrix, (sourceImage.shape[1], sourceImage.shape[0]))
    # plt.imshow(cv2.cvtColor(imageTransformed, cv2.COLOR_BGR2RGB))
    #
    # plt.show()
    cameraOffsetDegrees = 30
    cameraFocalHeight = 1  # D
    cameraFocalLength = math.sqrt(3)  # F
    projectionPlaneDistanceFromCenter = 10  # δ
    cameraPosition, cameraForwardVector, cameraFocalHeight, cameraFocalLength, projectionPlaneDistanceFromCenter = getStartingConditions(cameraOffsetDegrees, cameraFocalHeight, cameraFocalLength, projectionPlaneDistanceFromCenter)
    transformationMatrix = transformationMatrixMaker(cameraPosition, cameraForwardVector, cameraFocalHeight, cameraFocalLength, projectionPlaneDistanceFromCenter, (460, 341))
    print(np.matrix(transformationMatrix))

    # calculate output image dimensions based on focal height, focal length, and projection plane distance from center of camera
    # after transformation, the image will be centered at the origin
    # so the output image will be 2 * focal height by 2 * focal length
    # sourceImage = Image.open("test2.jpg")
    # width, height = sourceImage.size
    # imageTransformed =  sourceImage.transform((2 * int(width), 2 * int(height)), Image.AFFINE, data=transformationMatrix.flatten(), resample=Image.NEAREST)
    # plt.figure(figsize=(5, 5))
    # plt.imshow(np.asarray(imageTransformed))

    sourceImage = cv2.imread("test2.jpg")
    # convert transformationMatrix into a 2x3 matrix
    # transformationMatrix = np.float32(np.reshape(transformationMatrix[:2], (2, 3)))

    # flip transformationMatrix along the y-axis
    sourceImage = cv2.flip(sourceImage, 1)

    imageTransformed = cv2.warpPerspective(sourceImage, transformationMatrix, (sourceImage.shape[1], sourceImage.shape[0]))

    plt.imshow(cv2.cvtColor(imageTransformed, cv2.COLOR_BGR2RGB))

    plt.show()


def getStartingConditions(cameraOffsetDegrees: float, cameraFocalHeight: float, cameraFocalLength: float, projectionPlaneDistanceFromCenter: float):
    radians = math.radians(cameraOffsetDegrees)
    cameraPosition = (math.cos(radians), math.sin(radians), 0)
    cameraForwardVector = (math.cos(radians), math.sin(radians), 0)
    # cameraFocalHeight = 1  # D
    # cameraFocalLength = math.sqrt(3)  # F
    # projectionPlaneDistanceFromCenter = 10  # δ
    return cameraPosition, cameraForwardVector, cameraFocalHeight, cameraFocalLength, projectionPlaneDistanceFromCenter


def getTransformedImage(transformationMatrix, sourceImage):
    # convert transformationMatrix into a 2x3 matrix
    # transformationMatrix = np.float32(np.reshape(transformationMatrix[:2], (2, 3)))
    imageTransformed = cv2.warpPerspective(sourceImage, transformationMatrix, (sourceImage.shape[1], sourceImage.shape[0]))
    return imageTransformed


def transformationMatrixMaker(
        cameraPosition: [],
        cameraForwardVector: [],
        cameraFocalHeight: float,
        cameraFocalLength: float,
        projectionPlaneDistanceFromCenter: float,
        imageDimensions: (int, int) # width, height
) -> []:
    # first construct camera bounding corners
    focalAngle: float = math.atan(cameraFocalHeight / cameraFocalLength)
    # because we're initially facing in the (1, 0, 0) direction
    # we already know the vectors on the top and left bounding planes are going
    # to be expressed as
    # Camera top plane vector = [cos(focalAngle), 0, sin(focalAngle)]
    # Camera left plane vector = [cos(focalAngle), sin(focalAngle), 0]
    cameraPositiveZPlaneVector = [math.cos(focalAngle), 0, math.sin(focalAngle)] # Fy
    cameraPositiveYPlaneVector = [math.cos(focalAngle), math.sin(focalAngle), 0] # Fx
    # going counterclockwise starting in quadrant 1 from the view of the camera
    topRightCornerVector: [] = np.array(cameraPositiveYPlaneVector) - 2 * (np.array(cameraPositiveYPlaneVector) - np.array([math.sqrt(2) / 2, 0, 0])) + (np.array(cameraPositiveZPlaneVector) - np.array([math.sqrt(2) / 2, 0, 0]))
    bottomRightCornerVector: [] = np.array(cameraPositiveYPlaneVector) - 2 * (np.array(cameraPositiveYPlaneVector) - np.array([math.sqrt(2) / 2, 0, 0])) - (np.array(cameraPositiveZPlaneVector) - np.array([math.sqrt(2) / 2, 0, 0]))
    bottomLeftCornerVector: [] = np.array(cameraPositiveYPlaneVector) - (np.array(cameraPositiveZPlaneVector) - np.array([math.sqrt(2) / 2, 0, 0]))
    topLeftCornerVector: [] = np.array(cameraPositiveYPlaneVector) + (np.array(cameraPositiveZPlaneVector) - np.array([math.sqrt(2) / 2, 0, 0]))
    # topRightCornerVector = topRightCornerVector / np.sqrt(topRightCornerVector.dot(topRightCornerVector))
    # bottomRightCornerVector = bottomRightCornerVector / np.sqrt(bottomRightCornerVector.dot(bottomRightCornerVector))
    # bottomLeftCornerVector = bottomLeftCornerVector / np.sqrt(bottomLeftCornerVector.dot(bottomLeftCornerVector))
    # topLeftCornerVector = topLeftCornerVector / np.sqrt(topLeftCornerVector.dot(topLeftCornerVector))

    bottomRightCornerVector = np.copy(topRightCornerVector)
    bottomRightCornerVector[2] = -topRightCornerVector[2]
    topLeftCornerVector = np.copy(bottomLeftCornerVector)
    topLeftCornerVector[2] = -bottomLeftCornerVector[2]

    # projectedLines = [[1, 0, 0], topRightCornerVector, topLeftCornerVector, bottomLeftCornerVector, bottomRightCornerVector]
    # finalABPositions = []
    # for projectedLineVector in projectedLines:
    #     lineParameter = (projectionPlaneDistanceFromCenter - 1) / projectedLineVector[0]
    #     a = cameraPosition[1] + lineParameter * projectedLineVector[1]
    #     b = cameraPosition[2] + lineParameter * projectedLineVector[2]
    #     finalABPositions += [[a, b]]
    finalABPositions = [[0,0],[1,1],[-1,1],[-1,-1],[1,-1]]


    # projectedLines = [cameraForwardVector, topRightCornerVector, topLeftCornerVector, bottomLeftCornerVector, bottomRightCornerVector]
    # # print projected lines so that they can be copy pasted into 3D desmos
    # for projectedLineVector in projectedLines:
    #     print("(t * %.4f, t * %.4f, t * %.4f)" % (projectedLineVector[0], projectedLineVector[1], projectedLineVector[2]))
    # print()

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
    cameraPositiveYPlaneVector = np.matmul(zRotationMatrix, cameraPositiveYPlaneVector)

    # find intersection with delta plane
    # lineParameter = (projectionPlaneDistanceFromCenter - cameraPosition[0]) / (cameraForwardVector[0])
    # sticking everything in an array for convenience
    projectedLines = [cameraForwardVector, topRightCornerVector, topLeftCornerVector, bottomLeftCornerVector, bottomRightCornerVector]
    finalCDPositions = []
    for projectedLineVector in projectedLines:
        lineParameter = (projectionPlaneDistanceFromCenter - cameraPosition[0]) / projectedLineVector[0]
        c = cameraPosition[1] + lineParameter * projectedLineVector[1]
        d = cameraPosition[2] + lineParameter * projectedLineVector[2]
        finalCDPositions += [[c, d]]


    # print projected lines so that they can be copied and pasted into 3D desmos
    # for projectedLineVector in projectedLines:
    #     print("(t * %.4f, t * %.4f, t * %.4f)" % (projectedLineVector[0], projectedLineVector[1], projectedLineVector[2]))
    # print()

    # use this to convert the intersection point between the projected line and the focal plane
    # converting that into a vector, where we can take the bottom 2 coordinates as the image space coordinate
    # preTransformationPlaneImageSpaceTransformation = np.transpose([
    #     cameraForwardVector,
    #     -np.cross(cameraForwardVector, [0, 0, 1]),
    #     [0, 0, 1]
    # ])
    #
    # finalABPositions = []
    # for projectedLineVector in projectedLines:
    #     # get rid of first coordinate to get a and b
    #     # a is horizontal, b is vertical
    #     finalABPositions += [np.matmul(preTransformationPlaneImageSpaceTransformation, projectedLineVector)]
    # cameraSpaceOrigin = projectionPlaneDistanceFromCenter + postTransformationImageCoordinates[0]
    # worldBoundingBoxCorners = []
    # for projectedLineVector in projectedLines:
    #     cornerVectorLineParameter = (cameraForwardVector * cameraPosition - cameraForwardVector * cameraSpaceOrigin) / (cameraForwardVector * projectedLineVector) # t
    #     cameraBoundingBoxIntersection = cameraPosition + cornerVectorLineParameter * projectedLineVector
    # finalCDPositions = []
    # for coordinate in postTransformationImageCoordinates:
    #     finalCDPositions.append((projectionPlaneDistanceFromCenter, coordinate[0] - postTransformationImageCoordinates[0][0], coordinate[1] - postTransformationImageCoordinates[0][1]))

    # translate all points in AB and CD so that they're centered at the origin
    # use image dimensions to scale the points
    #
    # imageHeight = abs(finalCDPositions[2][1] - finalCDPositions[3][1])
    # imageResolutionHeight = imageDimensions[1] / imageHeight
    #
    # for i in range(len(finalCDPositions)):
    #     finalCDPositions[i][0] *= imageResolutionHeight
    #     finalCDPositions[i][1] *= imageResolutionHeight
    #
    # CDCenterX = finalCDPositions[1][0]
    # CDCenterY = finalCDPositions[3][1]
    #
    # for i in range(len(finalCDPositions)):
    #     finalCDPositions[i][0] -= CDCenterX
    #     finalCDPositions[i][1] -= CDCenterY
    #
    # # ABCenter = finalABPositions[3]
    # # ABTopRightDestination = [finalABPositions[1][0] - ABCenter[0], finalABPositions[1][1] - ABCenter[1]]
    # # xScalingFactor = finalCDPositions[1][0] / ABTopRightDestination[0]
    # imageHeight = abs(finalABPositions[1][1] - finalABPositions[4][1])
    # imageResolutionHeight = imageDimensions[1] / imageHeight
    #
    #
    #
    # for i in range(len(finalABPositions)):
    #     finalABPositions[i] = [finalABPositions[i][0] * imageResolutionHeight, finalABPositions[i][1] * imageResolutionHeight]
    #
    # ABCenter = finalABPositions[3]
    # for i in range(len(finalABPositions)):
    #     finalABPositions[i] = [finalABPositions[i][0] - ABCenter[0], finalABPositions[i][1] - ABCenter[1]]

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


    for point in finalABPositions[1:]:
        print("(%.4f, %.4f)" % (point[0], point[1]))
    print()
    for point in finalCDPositions[1:]:
        print("(%.4f, %.4f)" % (point[0], point[1]))
    print()



        # finalABPositions[i] = [x * (math.sqrt(projectionPlaneDistanceFromCenter**2 + CDCenter[0]**2 + CDCenter[1]**2) - 1) / (projectionPlaneDistanceFromCenter - 1) for x in finalABPositions[i]]

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
    return transformationMatrix


if __name__ == '__main__':
    main()