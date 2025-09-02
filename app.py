import cv2
import numpy as np
import time

# Start video capture
cap = cv2.VideoCapture(0)

# Wait for the camera to warm up
print("Camera starting... please wait 5 seconds and move out of the frame.")
time.sleep(5)

# Capture the background (when no person is in front)
for i in range(30):  # take multiple frames for a stable background
    ret, background = cap.read()
background = np.flip(background, axis=1)

# Define the color range for detecting red cloak
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame
    frame = np.flip(frame, axis=1)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    # Improve mask
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # Invert mask
    inverse_mask = cv2.bitwise_not(red_mask)

    # Apply masks
    res1 = cv2.bitwise_and(frame, frame, mask=inverse_mask)
    res2 = cv2.bitwise_and(background, background, mask=red_mask)

    # Final output
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    cv2.imshow("Invisibility Cloak", final_output)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
