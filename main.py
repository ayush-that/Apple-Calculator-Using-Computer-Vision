import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import handTrack as ht

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

while True:
    success, img = cap.read()
    img = cv.flip(img, 1)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList)

        # Tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()
        # print(fingers)

        if fingers[1] and fingers[2]:
            cv.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv.FILLED)
            print("Selection Mode")

            if y1 < 125:
                if 250 < x1 < 450:
                    asset = overlayList[0]
                elif 550 < x1 < 750:
                    asset = overlayList[1]
                elif 800 < x1 < 950:
                    asset = overlayList[2]
                elif 1050 < x1 < 1200:
                    asset = overlayList[3]

        if fingers[1] and fingers[2] == False:
            cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
            print("Drawing Mode")

    # Setting up the header image
    img[0:125, 0:1280] = asset
    cv.imshow("Image", img)
    cv.waitKey(1)
