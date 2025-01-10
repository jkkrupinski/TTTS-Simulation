import cv2 

from segmenter import Segmenter


cap = cv2.VideoCapture(2)  # TO ADJUST  lsusb   ls /dev/video*

    

if not cap.isOpened():
    print("Error: Could not access the camera.")


# Set camera properties (optional)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 620)  # Set width


# seg = Segmenter(
#     [
#         "TTTSNet_model-fold-0.pt",
#         "TTTSNet_model-fold-1.pt",
#         "TTTSNet_model-fold-2.pt",
#         "TTTSNet_model-fold-3.pt",
#         "TTTSNet_model-fold-4.pt",
#         "TTTSNet_model-fold-5.pt",
#     ]
# )

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    height, width, _ = frame.shape

    fov_ratio = 0.2 #0.145

    new_width = int(width * fov_ratio)
    new_height = int(height * fov_ratio)

    # Calculate the center crop coordinates
    x_center = width // 2
    y_center = height // 2
    x_start = x_center - new_width // 2
    y_start = y_center - new_height // 2
    x_end = x_center + new_width // 2
    y_end = y_center + new_height // 2
    # Crop the frame
    cropped_frame = frame[y_start:y_end, x_start+12:x_end-12]

    # cropped_frame = frame[0:480,70:550]

    resized_frame = cv2.resize(cropped_frame, (256, 256)) 

    cv2.imshow('USB Camera', resized_frame)

    resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)

    # out = seg(resized_frame)

    # cv2.imshow('Out', out)

   

    # If frame is read correctly, ret is True
    if not ret:
        print("Error: Couldn't read frame.")
        break

    # Display the resulting frame

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()