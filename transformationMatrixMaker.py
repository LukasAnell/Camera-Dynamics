import math
from numbers import Real
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
    result = transformationMatrixMaker(cameraPosition, cameraForwardVector, cameraFocalHeight, cameraFocalLength, projectionPlaneDistanceFromCenter)
    print(np.matrix(result))


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

    print(topRightCornerVector)
    print(bottomRightCornerVector)
    print(bottomLeftCornerVector)
    print(topLeftCornerVector)
    print()

    projectedLines = [[1, 0, 0], topRightCornerVector, topLeftCornerVector, bottomLeftCornerVector,
                      bottomRightCornerVector, cameraPositiveYPlaneVector]
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
    print(topRightCornerVector)
    print(bottomRightCornerVector)
    print(bottomLeftCornerVector)
    print(topLeftCornerVector)
    print()
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
    for i in range(6):
        finalABPositions[i][0] -= ABCenter[0]
        finalABPositions[i][1] -= ABCenter[1]
        finalCDPositions[i][0] -= CDCenter[0]
        finalCDPositions[i][1] -= CDCenter[1]

    x11 = Real('x11')
    x12 = Real('x12')
    x13 = Real('x13')
    x21 = Real('x21')
    x22 = Real('x22')
    x23 = Real('x23')
    x31 = Real('x31')
    x32 = Real('x32')
    x33 = Real('x33')
    equations = []
    for i in range(6):
        currentCD = finalCDPositions[i]
        currentAB = finalABPositions[i]
        equations += [
            currentCD[0] == (currentAB[0] * x11 + currentAB[1] * x12 + x13) / (currentAB[0] * x31 + currentAB[1] * x32 + x33),
            currentCD[1] == (currentAB[0] * x21 + currentAB[1] * x22 + x23) / (currentAB[0] * x31 + currentAB[1] * x32 + x33)
        ]
    solver = Solver()
    solver.add(equations)
    if solver.check() == sat:
        model = solver.model()
        solution = [
            [model[x11].as_decimal(6), model[x12].as_decimal(6), model[x13].as_decimal(6)],
            [model[x21].as_decimal(6), model[x22].as_decimal(6), model[x23].as_decimal(6)],
            [model[x31].as_decimal(6), model[x32].as_decimal(6), model[x33].as_decimal(6)]
        ]
        return solution
    else:
        print("No solution found")
        return None


if __name__ == '__main__':
    main()