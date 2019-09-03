import numpy as np
import cv2

PATH_TO_VIDEO = '../resources/valve_video.avi'


def main():
    cap = cv2.VideoCapture(PATH_TO_VIDEO)

    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:

            # Display the resulting frame
            cv2.imshow('frame', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
