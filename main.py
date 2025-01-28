import os

import cv2
import numpy as np


def main():
    path = R"spinning rat.mp4"
    cap = cv2.VideoCapture(path)
    while not cap.isOpened():
        cap = cv2.VideoCapture(path)
        cv2.waitKey(1000)
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 60.0, (int(cap.get(3)),int(cap.get(4))))

    frameNumber = -1
    ret, frame = cap.read()
    while ret and frameNumber <= cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frameNumber += 1
        ret, frame = cap.read()
        # do something to frame
        frame = cv2.flip(frame, 1)
        # write frame to output
        out.write(frame)

        if frame is not None and not frame.size == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    os.startfile('output.mp4')
    print("hi")


if __name__ == '__main__':
    main()