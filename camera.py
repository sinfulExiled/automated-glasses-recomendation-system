import os
from datetime import timedelta
from imutils.video import WebcamVideoStream
import face_recognition
import imutils
import numpy as np
import cv2
import pickle
import tensorflow as tf

import dlib
from imutils import face_utils

from watermarking import watermarking


class VideoCamera(object):
    def __init__(self):

        self.faceshapefind = None
        self.faceshapeangle = None
        self.stream = WebcamVideoStream(src=0).start()
        self.sendimg = self.stream.read()


    def __del__(self):
        self.stream.stop()

    def get_frame(self, makesave, imageformat):
        print(makesave)
        print(imageformat)

        imgOrignal = self.stream.read()
        NoFilter = self.stream.read()

        font = cv2.FONT_HERSHEY_SIMPLEX

        print("loading facial landmark predictor...")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # SETUP THE VIDEO CAMERA

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

        # PROCESS IMAGE
        img = np.asarray(NoFilter)
        img = cv2.resize(img, (32, 32))
        img = preprocessing(img)
        img = img.reshape(1, 32, 32, 1)
        cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        # PREDICT IMAGE
        predictions = model.predict(img)
        classIndex = model.predict_classes(img)
        probabilityValue = np.amax(predictions)

        if probabilityValue > 0.55:
            # print(getCalssName(classIndex))
            cv2.putText(imgOrignal, str(classIndex) + " " + str(getCalssName(classIndex)), (120, 35), font, 0.75,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255),
                        2,
                        cv2.LINE_AA)

        gray = cv2.cvtColor(imgOrignal, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayayscale frame
        rects = detector(gray, 0)

        # loopop over found faces
        for rect in rects:
            shape = predictor(imgOrignal, rect)
            shape = face_utils.shape_to_np(shape)

            eyeLeftSide = 0
            eyeRightSide = 0
            eyeTopSide = 0
            eyeBottomSide = 0

            for (i, (x, y)) in enumerate(shape):
                if (i + 1) == 37:
                    eyeLeftSide = x - 40
                if (i + 1) == 38:
                    eyeTopSide = y - 30
                if (i + 1) == 46:
                    eyeRightSide = x + 40
                if (i + 1) == 48:
                    eyeBottomSide = y + 30

            eyesWidth = eyeRightSide - eyeLeftSide
            if eyesWidth < 0:
                eyesWidth = eyesWidth * -1

            if classIndex == 0:
                glass = cv2.imread("Resources/glass5.png", -1)
                print("Heart")

            if classIndex == 1:
                glass = cv2.imread("Resources/glass2.png", -1)
                print("Oblong")

            if classIndex == 2:
                glass = cv2.imread("Resources/glass3.png", -1)
                print("Oval")

            if classIndex == 3:
                glass = cv2.imread("Resources/glass2.png", -1)
                print("Round")

            if classIndex == 4:
                glass = cv2.imread("Resources/glass.png", -1)
                print("Square")

            fitedGlass = imutils.resize(glass, width=eyesWidth)
            imgOrignal = watermarking(imgOrignal, fitedGlass, x=eyeLeftSide, y=eyeTopSide)

        ret, jpeg = cv2.imencode('.jpg', imgOrignal)
        # if makesave == "Save":
        #
        #     location = "static/" + "d" + "." + imageformat
        #     print(location)
        #     cv2.imwrite(location, imgOrignal)
        #     makesave = "noSave"
        #     from form_data import cnn_predict
        #     return
        data = []
        data.append(jpeg.tobytes())
        return data

    def save(self, imageformat):
        import time
        timestr = time.strftime("%Y%m%d-%H%M%S")
        imageNames = []
        x = 0
        for x in range(4):
            applyfilterimg = self.applyfilter()
            location = "static/saves"
            imgname = timestr + str(x) + "." + imageformat
            print(location)
            imageNames.append(imgname)
            print(imageNames)
            cv2.imwrite(os.path.join(location, imgname), applyfilterimg)
            x += 1

        return imageNames

    def savekmeans(self, imageformat):
        import time
        timestr = time.strftime("%Y%m%d-%H%M%S")
        imageNames = []
        kmeansfilter = self.kmeansfilter()
        print(self.faceshapefind)
        print("zzzzzzzzzzzzzzz")
        location = "static/saves"
        imgname = timestr + "." + imageformat
        print(location)
        imageNames.append(imgname)
        imageNames.append(self.faceshapefind)
        imageNames.append(self.faceshapeangle)
        print(imageNames)
        cv2.imwrite(os.path.join(location, imgname), kmeansfilter)
        return imageNames

    def applyfilter(self):
        imgOrignal = self.stream.read()
        NoFilter = self.stream.read()

        font = cv2.FONT_HERSHEY_SIMPLEX

        print("loading facial landmark predictor...")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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

        img = np.asarray(NoFilter)
        img = cv2.resize(img, (32, 32))
        img = preprocessing(img)
        img = img.reshape(1, 32, 32, 1)
        cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        predictions = model.predict(img)
        classIndex = model.predict_classes(img)
        probabilityValue = np.amax(predictions)

        if probabilityValue > 0.55:
            cv2.putText(imgOrignal, str(classIndex) + " " + str(getCalssName(classIndex)), (120, 35), font, 0.75,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255),
                        2,
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

            for (i, (x, y)) in enumerate(shape):
                if (i + 1) == 37:
                    eyeLeftSide = x - 40
                if (i + 1) == 38:
                    eyeTopSide = y - 30
                if (i + 1) == 46:
                    eyeRightSide = x + 40
                if (i + 1) == 48:
                    eyeBottomSide = y + 30

            eyesWidth = eyeRightSide - eyeLeftSide
            if eyesWidth < 0:
                eyesWidth = eyesWidth * -1

            if classIndex == 0:
                glass = cv2.imread("Resources/glass5.png", -1)
                print("Heart")

            if classIndex == 1:
                glass = cv2.imread("Resources/glass2.png", -1)
                print("Oblong")

            if classIndex == 2:
                glass = cv2.imread("Resources/glass3.png", -1)
                print("Oval")

            if classIndex == 3:
                glass = cv2.imread("Resources/glass2.png", -1)
                print("Round")

            if classIndex == 4:
                glass = cv2.imread("Resources/glass.png", -1)
                print("Square")

            fitedGlass = imutils.resize(glass, width=eyesWidth)
            imgOrignal = watermarking(imgOrignal, fitedGlass, x=eyeLeftSide, y=eyeTopSide)

        return imgOrignal


    def kmeansfilter(self):
        import numpy as np
        import cv2
        import dlib
        from sklearn.cluster import KMeans

        image = self.stream.read()

        face_cascade_path = "D:\Open CV CLS\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml"

        predictor_path = "shape_predictor_68_face_landmarks.dat"

        faceCascade = cv2.CascadeClassifier(face_cascade_path)

        predictor = dlib.shape_predictor(predictor_path)

        image = cv2.resize(image, (500, 500))
        original = image.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gauss = cv2.GaussianBlur(gray, (3, 3), 0)

        faces = faceCascade.detectMultiScale(
            gauss,
            scaleFactor=1.05,
            minNeighbors=1,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        print("found {0} faces!".format(len(faces)))

        for (x, y, w, h) in faces:

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

            detected_landmarks = predictor(image, dlib_rect).parts()

            landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])
            print(landmarks)

            landmark = image.copy()
            for idx, point in enumerate(landmarks):
                pos = (point[0, 0], point[0, 1])

                cv2.putText(landmark, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                            color=(0, 0, 255))

                cv2.circle(landmark, pos, 3, color=(0, 255, 255))

        results = original.copy()

        for (x, y, w, h) in faces:
            cv2.rectangle(results, (x, y), (x + w, y + h), (0, 255, 0), 2)

            temp = original.copy()

            forehead = temp[y:y + int(0.25 * h), x:x + w]
            rows, cols, bands = forehead.shape
            X = forehead.reshape(rows * cols, bands)

            kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
            y_kmeans = kmeans.fit_predict(X)
            for i in range(0, rows):
                for j in range(0, cols):
                    if y_kmeans[i * cols + j] == True:
                        forehead[i][j] = [255, 255, 255]
                    if y_kmeans[i * cols + j] == False:
                        forehead[i][j] = [0, 0, 0]

            forehead_mid = [int(cols / 2), int(rows / 2)]
            lef = 0
            pixel_value = forehead[forehead_mid[1], forehead_mid[0]]
            for i in range(0, cols):

                if forehead[forehead_mid[1], forehead_mid[0] - i].all() != pixel_value.all():
                    lef = forehead_mid[0] - i
                    break;
            left = [lef, forehead_mid[1]]
            rig = 0
            for i in range(0, cols):
                if forehead[forehead_mid[1], forehead_mid[0] + i].all() != pixel_value.all():
                    rig = forehead_mid[0] + i
                    break;
            right = [rig, forehead_mid[1]]

        line1 = np.subtract(right + y, left + x)[0]

        cv2.line(results, tuple(x + left), tuple(y + right), color=(0, 255, 0), thickness=2)
        cv2.putText(results, ' Line 1', tuple(x + left), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 255, 0),
                    thickness=2)
        cv2.circle(results, tuple(x + left), 5, color=(255, 0, 0), thickness=-1)
        cv2.circle(results, tuple(y + right), 5, color=(255, 0, 0), thickness=-1)

        linepointleft = (landmarks[1, 0], landmarks[1, 1])
        linepointright = (landmarks[15, 0], landmarks[15, 1])
        line2 = np.subtract(linepointright, linepointleft)[0]
        print(line2, "line2")
        cv2.line(results, linepointleft, linepointright, color=(0, 255, 0), thickness=2)
        cv2.putText(results, ' Line 2', linepointleft, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 255, 0),
                    thickness=2)
        cv2.circle(results, linepointleft, 5, color=(255, 0, 0), thickness=-1)
        cv2.circle(results, linepointright, 5, color=(255, 0, 0), thickness=-1)

        linepointleft = (landmarks[3, 0], landmarks[3, 1])
        linepointright = (landmarks[13, 0], landmarks[13, 1])
        line3 = np.subtract(linepointright, linepointleft)[0]
        cv2.line(results, linepointleft, linepointright, color=(0, 255, 0), thickness=2)
        cv2.putText(results, ' Line 3', linepointleft, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 255, 0),
                    thickness=2)
        cv2.circle(results, linepointleft, 5, color=(255, 0, 0), thickness=-1)
        cv2.circle(results, linepointright, 5, color=(255, 0, 0), thickness=-1)

        linepointbottom = (landmarks[8, 0], landmarks[8, 1])
        linepointtop = (landmarks[8, 0], y)
        line4 = np.subtract(linepointbottom, linepointtop)[1]
        cv2.line(results, linepointtop, linepointbottom, color=(0, 255, 0), thickness=2)
        cv2.putText(results, ' Line 4', linepointbottom, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 255, 0),
                    thickness=2)
        cv2.circle(results, linepointtop, 5, color=(255, 0, 0), thickness=-1)
        cv2.circle(results, linepointbottom, 5, color=(255, 0, 0), thickness=-1)
        print(line1, line2, line3, line4)

        similarity = np.std([line1, line2, line3])
        print("similarity=", similarity)

        # we use arcustangens for angle calculation
        ax, ay = landmarks[3, 0], landmarks[3, 1]
        print('ax, ay=', ax, ay)
        bx, by = landmarks[4, 0], landmarks[4, 1]
        cx, cy = landmarks[5, 0], landmarks[5, 1]
        dx, dy = landmarks[6, 0], landmarks[6, 1]
        import math
        from math import degrees

        alpha0 = math.atan2(cy - ay, cx - ax)
        alpha1 = math.atan2(dy - by, dx - bx)
        alpha = alpha1 - alpha0
        angle = abs(degrees(alpha))

        angle = 180 - angle
        classIndex = 0
        faceshape = "null";
        for i in range(1):
            if similarity < 10:
                if angle < 160:
                    faceshape = "Squared"
                    classIndex = 0
                    break
                else:
                    faceshape = "Round"
                    classIndex = 1
                    break
            if line3 > line1:
                if angle < 160:
                    faceshape = "Heart"
                    classIndex = 2
                    break

            if line4 > line2:
                if angle < 160:
                    faceshape = "Oval"
                    classIndex = 3
                    break;
                else:
                    faceshape = "Oblong"
                    classIndex = 4
                    break;
            print("Error")
        detector = dlib.get_frontal_face_detector()

        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(original, rect)
            shape = face_utils.shape_to_np(shape)

            eyeLeftSide = 0
            eyeRightSide = 0
            eyeTopSide = 0
            eyeBottomSide = 0

            for (i, (x, y)) in enumerate(shape):
                if (i + 1) == 37:
                    eyeLeftSide = x - 40
                if (i + 1) == 38:
                    eyeTopSide = y - 30
                if (i + 1) == 46:
                    eyeRightSide = x + 40
                if (i + 1) == 48:
                    eyeBottomSide = y + 30

            eyesWidth = eyeRightSide - eyeLeftSide
            if eyesWidth < 0:
                eyesWidth = eyesWidth * -1

            if classIndex == 0:
                glass = cv2.imread("Resources/glass5.png", -1)
                print("Heart")

            if classIndex == 1:
                glass = cv2.imread("Resources/glass2.png", -1)
                print("Oblong")

            if classIndex == 2:
                glass = cv2.imread("Resources/glass3.png", -1)
                print("Oval")

            if classIndex == 3:
                glass = cv2.imread("Resources/glass2.png", -1)
                print("Round")

            if classIndex == 4:
                glass = cv2.imread("Resources/glass.png", -1)
                print("Square")

            fitedGlass = imutils.resize(glass, width=eyesWidth)
            original = watermarking(original, fitedGlass, x=eyeLeftSide, y=eyeTopSide)

        self.faceshapefind = faceshape
        self.faceshapeangle = angle
        output = np.concatenate((original, results), axis=1)
        return output