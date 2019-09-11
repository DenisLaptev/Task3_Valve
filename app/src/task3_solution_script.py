import numpy as np
import cv2

PATH_TO_VIDEO = '../resources/valve_video.avi'

PATH_TO_SAVE_VIDEO_THRESHOLD = '../output/valve_video_threshold.avi'
PATH_TO_SAVE_VIDEO_DEBUG = '../output/valve_video_debug.avi'
PATH_TO_SAVE_VIDEO_RESULT = '../output/valve_video_result.avi'

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_thresh = cv2.VideoWriter(PATH_TO_SAVE_VIDEO_THRESHOLD, fourcc, 20.0, (640, 480), 0)
out_debug = cv2.VideoWriter(PATH_TO_SAVE_VIDEO_DEBUG, fourcc, 20.0, (640, 480), 0)
out_result = cv2.VideoWriter(PATH_TO_SAVE_VIDEO_RESULT, fourcc, 20.0, (640, 480), 0)


def modify_frame(frame_initial):
    # cvt to gray
    frame_gray = cv2.cvtColor(frame_initial, cv2.COLOR_BGR2GRAY)

    # bilateral filter
    frame_gray_filtered = cv2.bilateralFilter(frame_gray, 17, 150, 150)

    # threshold
    _,frame_threshold = cv2.threshold(frame_gray_filtered, 30, 255, cv2.THRESH_BINARY)
    return frame_threshold


def get_large_contours(contours):
    large_contours = []
    number_of_contours = len(contours)
    if (number_of_contours > 0):
        max_area_new_contour = contours[0]
        max_area = cv2.contourArea(contours[0])
        for i in range(number_of_contours):
            current_cnt = contours[i]
            cnt_area = cv2.contourArea(current_cnt)
            if cnt_area > max_area:
                max_area_new_contour = current_cnt
        large_contours.append(max_area_new_contour)
    return large_contours



def main():
    cap = cv2.VideoCapture(PATH_TO_VIDEO)

    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame_initial = cap.read()

        if ret == True:

            frame_initial_copy = frame_initial.copy()

            frame_modified = modify_frame(frame_initial)

            contours_ff_filt, hierarchy = cv2.findContours(frame_modified, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            new_contours = []
            for cnt in contours_ff_filt:
                # area
                area = cv2.contourArea(cnt)

                # perimeter
                perimeter = cv2.arcLength(cnt, True)

                # area/perimeter <= perimeter /(4*Pi) (=for circle)
                # we asume that area/perimeter > perimeter/ 18

                x, y, w, h = cv2.boundingRect(cnt)

                # aspect ratio = w/h
                aspect_ratio = float(w) / h

                M = cv2.moments(cnt)
                if (M['m00'] == 0):
                    cx = int(M['m10'] / (M['m00'] + 0.000001))
                    cy = int(M['m01'] / (M['m00'] + 0.000001))
                else:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                # solidity =  (contour area) / (convex hull area)
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    solidity = float(area) / (hull_area + 0.000001)
                else:
                    solidity = float(area) / hull_area

                if perimeter > 250 and perimeter < 2000 \
                        and ((abs(cx - 323) < 100) and abs(cy - 349) < 100)\
                        and solidity > 0.5:  # \

                    new_contours.append(cnt)

            large_contours = get_large_contours(new_contours)
            for large_cnt in large_contours:
                x, y, w, h = cv2.boundingRect(large_cnt)
                cv2.rectangle(frame_initial_copy, (x, y), (x + w, y + h), (0, 0, 255), 5)


            cv2.drawContours(frame_initial_copy, large_contours, 0, (0, 255, 0), 5)
            cv2.drawContours(frame_initial, large_contours, 0, (0, 255, 0), 5)

            # saving output
            out_thresh.write(frame_modified)
            out_debug.write(frame_initial_copy)
            out_result.write(frame_initial)


            cv2.imshow('frame_initial_copy', frame_initial_copy)
            cv2.imshow('Result', frame_initial)
            cv2.imshow('frame_modified', frame_modified)

            key = cv2.waitKey(25)
            if key == 27:
                break

        else:
            break

    # When everything done, release the capture and VideoWriters
    cap.release()
    out_thresh.release()
    out_debug.release()
    out_result.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
