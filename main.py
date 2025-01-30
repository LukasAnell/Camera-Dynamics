import os
import cv2
import numpy as np


def main():
    path = R"spinning rat.mp4"

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    width = 402
    height = 360
    output = cv2.VideoWriter('output.mp4', fourcc, 60.0, (width * 3, height))

    cap = cv2.VideoCapture(path)
    cap2 = cv2.VideoCapture(path)
    cap3 = cv2.VideoCapture(path)

    frameNumber = -1
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    while ret and ret2 and ret3 and frameNumber <= cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frameNumber += 1
        ret, frame = cap.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()

        # do something to frame
        bothFrames = cv2.hconcat([frame, frame2, frame3])
        # write frame to output
        output.write(bothFrames)

        # if frame is not None and frame2 is not None and not frame.size == 0 and not frame2.size == 0:
        #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     # cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cap2.release()
    cap3.release()
    output.release()

    cv2.destroyAllWindows()
    os.startfile('output.mp4')
    print("hi")
    return


if __name__ == '__main__':
    main()