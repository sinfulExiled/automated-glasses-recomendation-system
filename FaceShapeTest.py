import imutils
import numpy as np
import cv2
import pickle
import tensorflow as tf

import dlib
from imutils import face_utils

from watermarking import watermarking

#############################################

frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.55  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

moustache = cv2.imread("Resources/moustache.png", cv2.IMREAD_UNCHANGED)
##############################################

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
# IMPORT THE TRANNIED MODEL
model = tf.keras.models.load_model('faceshape_model.h5')


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def getCalssName(classNo):
    if classNo == 0:
        return 'Face Shape Heart'
    elif classNo == 1:
        return 'Face Shape Oblong'
    elif classNo == 2:
        return 'Face Shape Oval'
    elif classNo == 3:
        return 'Face Shape Round'
    elif classNo == 4:
        return 'Face Shape Square'


def findInKMeans(imagepath):
    import FaceShapeKMeans
    FaceShapeKMeans.runMethod(imagepath)


#
# cam = cv2.VideoCapture(0)

img_counter = 0

while True:

    success, imgOrignal = cap.read()
    success2, NoFilter = cap.read()

    img = np.asarray(NoFilter)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    predictions = model.predict(img)
    print(predictions)
    classIndex = model.predict_classes(img)
    probabilityValue = np.amax(predictions)

    if probabilityValue > threshold:
        # print(getCalssName(classIndex))
        cv2.putText(imgOrignal, str(classIndex) + " " + str(getCalssName(classIndex)), (120, 35), font, 0.75,
                    (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2,
                    cv2.LINE_AA)

    gray = cv2.cvtColor(imgOrignal, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(imgOrignal, rect)
        shape = face_utils.shape_to_np(shape)

        eyeLeftSide = 0
        eyeRightSide = 0
        eyeTopSide = 0
        eyeBottomSide = 0

        moustacheLeftSide = 0
        moustacheRightSide = 0
        moustacheTopSide = 0
        moustacheBottomSide = 0

        for (i, (x, y)) in enumerate(shape):
            if (i + 1) == 37:
                eyeLeftSide = x - 40
            if (i + 1) == 38:
                eyeTopSide = y - 30
            if (i + 1) == 46:
                eyeRightSide = x + 40
            if (i + 1) == 48:
                eyeBottomSide = y + 30

            if (i + 1) == 2:
                moustacheLeftSide = x
                moustacheTopSide = y - 10
            if (i + 1) == 16:
                moustacheRightSide = x
            if (i + 1) == 9:
                moustacheBottomSide = y

        eyesWidth = eyeRightSide - eyeLeftSide
        if eyesWidth < 0:
            eyesWidth = eyesWidth * -1

        # add glasses

        if classIndex == 0:
            glass = cv2.imread("Resources/glass5.png", cv2.IMREAD_UNCHANGED)
            print("Heart")

        if classIndex == 1:
            glass = cv2.imread("Resources/glass2.png", cv2.IMREAD_UNCHANGED)
            print("Oblong")

        if classIndex == 2:
            glass = cv2.imread("Resources/glass3.png", cv2.IMREAD_UNCHANGED)
            print("Oval")

        if classIndex == 3:
            glass = cv2.imread("Resources/glass2.png", cv2.IMREAD_UNCHANGED)
            print("Round")

        if classIndex == 4:
            glass = cv2.imread("Resources/glass.png", cv2.IMREAD_UNCHANGED)
            print("Square")

        fitedGlass = imutils.resize(glass, width=eyesWidth)
        imgOrignal = watermarking(imgOrignal, fitedGlass, x=eyeLeftSide, y=eyeTopSide)

    cv2.imshow("Try On Glasses", imgOrignal)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Closing... Pressed Escape")
        break
    elif k % 256 == 32:

        img_name = "C:/Users/SinfulExiled/PycharmProjects/OpenCvTryOn/CapturedImages/opencv_frame_{}.jpg".format(
            img_counter)
        cv2.imwrite(img_name, NoFilter)
        print("{} written!".format(img_name))
        img_counter += 1
        findInKMeans(NoFilter)
