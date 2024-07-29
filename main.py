import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import handTrack as ht

brushThickness = 15
eraserThickness = 100

folderPath = "assets"
mylist = os.listdir(folderPath)
print(mylist)
overlayList = []

for imPath in mylist:
    image = cv.imread(f"{folderPath}/{imPath}")
    overlayList.append(image)

print(len(overlayList))
asset = overlayList[0]

drawColor = (255, 0, 255)

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = ht.handDetector(detectionCon=0.85)
xp, yp = 0, 0

imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img = cap.read()
    img = cv.flip(img, 1)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv.FILLED) # Selection Mode

            if y1 < 125:
                if 250 < x1 < 450:
                    asset = overlayList[0]
                    drawColor = (255, 0, 255)
                    cv.imwrite("saved_canvas.jpg", imgCanvas)
                    os.system("python app.py")
                elif 550 < x1 < 750:
                    asset = overlayList[1]
                    drawColor = (255, 0, 0)
                    cv.imwrite("saved_canvas.jpg", imgCanvas)
                elif 800 < x1 < 950:
                    asset = overlayList[2]
                    drawColor = (0, 255, 0)
                    cv.imwrite("saved_canvas.jpg", imgCanvas)
                elif 1050 < x1 < 1200:
                    asset = overlayList[3]
                    drawColor = (0, 0, 0)
                    cv.imwrite("saved_canvas.jpg", imgCanvas)

        if fingers[1] and not fingers[2]:
            cv.circle(img, (x1, y1), 15, drawColor, cv.FILLED) # Drawing Mode
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

    imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img, imgInv)
    img = cv.bitwise_or(img, imgCanvas)

    # Setting up the header image
    img[0:125, 0:1280] = asset
    img = cv.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv.imshow("Image", img)
    cv.imshow("Canvas", imgCanvas)
    cv.imshow("Inv", imgInv)
    cv.waitKey(1)
