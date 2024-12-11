import cv2
import numpy as np

# Define blue, red, and green color ranges in HSV
blue_lower = np.array([100, 150, 0], dtype=np.uint8)
blue_upper = np.array([140, 255, 255], dtype=np.uint8)

red_lower1 = np.array([0, 120, 70], dtype=np.uint8)
red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
red_lower2 = np.array([170, 120, 70], dtype=np.uint8)
red_upper2 = np.array([180, 255, 255], dtype=np.uint8)

green_lower = np.array([40, 70, 70], dtype=np.uint8)
green_upper = np.array([80, 255, 255], dtype=np.uint8)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask for blue color
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # Create mask for red color
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Create mask for green color
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # Combine all masks
    combined_mask = cv2.bitwise_or(cv2.bitwise_or(blue_mask, red_mask), green_mask)

    # Apply the combined mask to get only blue, red, and green parts
    colored_output = cv2.bitwise_and(frame, frame, mask=combined_mask)

    # Convert the masks to grayscale to find contours
    gray_mask = cv2.cvtColor(colored_output, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            # Draw rectangle around the detected object
            x, y, w, h = cv2.boundingRect(cnt)

            # Determine the color of the object by checking the mean value of the respective mask
            if cv2.mean(blue_mask[y:y+h, x:x+w])[0] > 0:
                color_name = "blue"
            elif cv2.mean(red_mask[y:y+h, x:x+w])[0] > 0:
                color_name = "red"
            elif cv2.mean(green_mask[y:y+h, x:x+w])[0] > 0:
                color_name = "green"
            else:
                color_name = "unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Print the center of the object and its color
            cx, cy = x + w // 2, y + h // 2
            print(f"Center of the {color_name} object: ({cx}, {cy})")

    # Show the original frame and the mask
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Colored Mask', colored_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
