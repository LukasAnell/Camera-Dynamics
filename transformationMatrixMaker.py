import math
from numbers import Real
import cv2
import numpy as np
from z3 import *

# assume center at (0, 0, 0)
radians = math.radians(15)
cameraPosition = (math.cos(radians), math.sin(radians), 0)
cameraForwardVector = (math.cos(radians), math.sin(radians), 0)
cameraFocalHeight = 1 # D
cameraFocalLength = math.sqrt(3) # F
projectionPlaneDistanceFromCenter = 5 # δ
# assume camera initially faces (1, 0, 0)
# and we will rotate the camera to face that once we
# make the corner vectors

def main():
    transformationMatrix = transformationMatrixMaker(cameraPosition, cameraForwardVector, cameraFocalHeight, cameraFocalLength, projectionPlaneDistanceFromCenter)
    print(np.matrix(transformationMatrix))
    # calculate output image dimensions based on focal height, focal length, and projection plane distance from center of camera
    # after transformation, the image will be centered at the origin
    # so the output image will be 2 * focal height by 2 * focal length
    sourceImage = cv2.imread("test2.jpg")
    outputImage = cv2.warpPerspective(sourceImage, transformationMatrix, (int(2 * cameraFocalLength), int(2 * cameraFocalHeight)))
    cv2.imshow("output", outputImage)


def transformationMatrixMaker(
        cameraPosition: [],
        cameraForwardVector: [],
        cameraFocalHeight: float,
        cameraFocalLength: float,
        projectionPlaneDistanceFromCenter: float
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

    projectedLines = [[1, 0, 0], topRightCornerVector, topLeftCornerVector, bottomLeftCornerVector, bottomRightCornerVector, cameraPositiveYPlaneVector]
    finalABPositions = []
    for projectedLineVector in projectedLines:
        lineParameter = (projectionPlaneDistanceFromCenter - cameraPosition[0]) / projectedLineVector[0]
        a = cameraPosition[1] + lineParameter * projectedLineVector[1]
        b = cameraPosition[2] + lineParameter * projectedLineVector[2]
        finalABPositions += [[a, b]]

    # compute the angle between cameras
    # a dot b = |a| * |b| cos(theta)
    # we are assuming magnitude of a and b are 1
    angleBetweenCameras = math.acos(np.dot(cameraForwardVector, [1, 0, 0]))
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
    projectedLines = [cameraForwardVector, topRightCornerVector, topLeftCornerVector, bottomLeftCornerVector, bottomRightCornerVector, cameraPositiveYPlaneVector]
    finalCDPositions = []
    for projectedLineVector in projectedLines:
        lineParameter = (projectionPlaneDistanceFromCenter - cameraPosition[0]) / projectedLineVector[0]
        c = cameraPosition[1] + lineParameter * projectedLineVector[1]
        d = cameraPosition[2] + lineParameter * projectedLineVector[2]
        finalCDPositions += [[c, d]]
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
    ABCenter = finalABPositions[0]
    CDCenter = finalCDPositions[0]
    for i in range(5):
        finalABPositions[i][0] -= ABCenter[0]
        finalABPositions[i][1] -= ABCenter[1]
        finalCDPositions[i][0] -= CDCenter[0]
        finalCDPositions[i][1] -= CDCenter[1]

    # use cv2.getPerspectiveTransform to get transformation matrix
    # AB is the source, CD is the destination
    # Ensure AB and CD are numpy arrays of type float32 and contain exactly 4 points
    AB = np.array(finalABPositions[1:-1], dtype=np.float32)
    CD = np.array(finalCDPositions[1:-1], dtype=np.float32)

    transformationMatrix = cv2.getPerspectiveTransform(AB, CD)
    return transformationMatrix


if __name__ == '__main__':
    main()