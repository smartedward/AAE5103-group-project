import net.mtcnn as mtcnn
import cv2
import numpy as np
if __name__ == '__main__':
    # threshold = [0.5, 0.6, 0.8]
    # img = cv2.imread('D:/pycharm/opencv/keras-face-recognition-master/face_dataset/obama.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # want_face = mtcnn.mtcnn()
    # rectangles = want_face.detectFace(img,threshold)
    # draw = img.copy()
    # for rectangle in rectangles:
    #
    #
    #     cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (0, 0, 255), 2)
    #
    #     for i in range(5, 15, 2):
    #         cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 1, (255, 0, 0), 4)
    # cv2.imshow('obama',draw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
