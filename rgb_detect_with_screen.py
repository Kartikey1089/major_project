import cv2
import numpy as np
from RPLCD.i2c import CharLCD

# Initialize LCD: Adjust i2c_expander, address, and other settings to match your setup
lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2)

# Function to detect color and return coordinates
def detect_color(frame, lower_bound, upper_bound, color_name):
    mask = cv2.inRange(frame, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Adjust this threshold as per your object size
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2  # Coordinates of the center of the rectangle
            
            # Draw rectangle around detected color and display name
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
            cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Return the coordinates for LCD display
            return cx, cy

    return None, None

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for red, green, and blue in HSV
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    lower_red_alt = np.array([170, 120, 70])
    upper_red_alt = np.array([180, 255, 255])

    lower_green = np.array([36, 100, 100])
    upper_green = np.array([86, 255, 255])

    lower_blue = np.array([94, 80, 2])
    upper_blue = np.array([126, 255, 255])

    # Detect red, green, and blue colors
    red_x, red_y = detect_color(hsv, lower_red, upper_red, 'Red')
    red_x_alt, red_y_alt = detect_color(hsv, lower_red_alt, upper_red_alt, 'Red')
    green_x, green_y = detect_color(hsv, lower_green, upper_green, 'Green')
    blue_x, blue_y = detect_color(hsv, lower_blue, upper_blue, 'Blue')

    # Check if any color was detected and display coordinates on the LCD
    if red_x is not None or red_x_alt is not None:
        lcd.clear()
        lcd.write_string(f"Red: {red_x or red_x_alt},{red_y or red_y_alt}")
    elif green_x is not None:
        lcd.clear()
        lcd.write_string(f"Green: {green_x},{green_y}")
    elif blue_x is not None:
        lcd.clear()
        lcd.write_string(f"Blue: {blue_x},{blue_y}")

    # Display the result in a window
    cv2.imshow("Color Detection", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
lcd.clear()
