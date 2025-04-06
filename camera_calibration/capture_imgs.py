import cv2 as cv
import os

# Create directory if it doesn't exist
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

output_dir = os.path.join(script_dir, 'checkerboard_imgs')

os.makedirs(output_dir, exist_ok=True)

# Open the camera
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

img_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to grayscale
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Display the grayscale frame
    cv.imshow("Camera - Press 's' to save, 'q' to quit", gray_frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('s'):  # Save image on 's' key press
        img_name = os.path.join(output_dir, f"image_{img_counter:04d}.png")
        cv.imwrite(img_name, gray_frame)
        print(f"Saved: {img_name}")
        img_counter += 1
    elif key == ord('q'):  # Quit on 'q' key press
        break

# Release the camera and close windows
cap.release()
cv.destroyAllWindows()