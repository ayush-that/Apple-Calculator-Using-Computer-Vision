from flask import Flask, render_template, Response, request, redirect, url_for
import cv2 as cv
import numpy as np
import os
import handTrack as ht
import google.generativeai as genai

app = Flask(__name__)

brushThickness = 15
eraserThickness = 100

folderPath = "assets"
mylist = os.listdir(folderPath)
overlayList = [cv.imread(f"{folderPath}/{imPath}") for imPath in mylist]

asset = overlayList[0]
drawColor = (0, 0, 255)

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = ht.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

genai.configure(api_key="")

def gen_frames():
    global xp, yp, asset, drawColor, imgCanvas

    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            img = cv.flip(img, 1)
            img = detector.findHands(img)
            lmList = detector.findPosition(img, draw=False)

            if len(lmList) != 0:
                x1, y1 = lmList[8][1:]
                x2, y2 = lmList[12][1:]

                fingers = detector.fingersUp()

                if fingers[1] and fingers[2]:
                    xp, yp = 0, 0
                    cv.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv.FILLED)

                    if y1 < 125:
                        if 550 < x1 < 750:
                            asset = overlayList[1]
                            drawColor = (255, 0, 0)
                            cv.imwrite("saved_canvas.jpg", imgCanvas)
                        elif 800 < x1 < 950:
                            asset = overlayList[2]
                            drawColor = (0, 0, 255)
                            cv.imwrite("saved_canvas.jpg", imgCanvas)
                        elif 1050 < x1 < 1200:
                            asset = overlayList[3]
                            drawColor = (0, 0, 0)
                            cv.imwrite("saved_canvas.jpg", imgCanvas)

                if fingers[1] and not fingers[2]:
                    cv.circle(img, (x1, y1), 15, drawColor, cv.FILLED)
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

            img[0:125, 0:1280] = asset
            img = cv.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

            ret, buffer = cv.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gemini')
def gemini():
    sample_file = genai.upload_file(path="saved_canvas.jpg", display_name="Maths Question")
    file = genai.get_file(name=sample_file.name)
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    response = model.generate_content([sample_file, "In the picture you have been provided a mathematics question/equation. Please solve it. First write the final solution then write the explanation. Give plain text response only."])
    return response.text

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
