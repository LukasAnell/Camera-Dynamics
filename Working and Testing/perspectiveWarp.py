import cv2
import numpy as np


def perspectiveWarp(frame, width, height):
    # for side cameras, needs to alter current frame to remove perspective warping due to the camera being at an angle
    # cameras are offset by 60 degrees
    # use to create transformation matrix to then use in cv2.warpPerspective()
    # for now, just return the frame
    transformationMatrix = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    frame = cv2.warpPerspective(frame, transformationMatrix, (width, height))
    return frame