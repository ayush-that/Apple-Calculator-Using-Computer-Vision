import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define the upper and lower bounds for the color to be detected (e.g., green color)
lower_bound = np.array([40, 100, 100])
upper_bound = np.array([80, 255, 255])

# Initialize an empty canvas
canvas = None

# Initialize previous x, y coordinates
prev_x, prev_y = 0, 0

while True:
    # Capture the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the color detection
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If a contour is found, get the coordinates
    if contours and len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cx, cy = x + w // 2, y + h // 2

        # Initialize the canvas if it's not already
        if canvas is None:
            canvas = np.zeros_like(frame)

        # If the previous coordinates are not (0, 0), draw a line
        if prev_x != 0 and prev_y != 0:
            canvas = cv2.line(canvas, (prev_x, prev_y), (cx, cy), (0, 255, 0), 5)

        # Update the previous coordinates
        prev_x, prev_y = cx, cy
    else:
        prev_x, prev_y = 0, 0

    # Combine the canvas and the frame
    combined = cv2.add(frame, canvas)

    # Display the result
    cv2.imshow('Air Canvas', combined)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
