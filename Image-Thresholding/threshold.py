import ctypes
import numpy as np
import cv2


# Load library
lib = ctypes.cdll.LoadLibrary('./thresholding-cuda/threshold.dll')


# Define argument types
lib.threshold_image.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_ubyte]

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

threshold = 127

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Ensure contiguous array
    gray = np.ascontiguousarray(gray)

    # Call your CUDA thresholding
    img_ptr = gray.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    lib.threshold_image(img_ptr, gray.size, threshold)

    # Display result
    cv2.imshow('CUDA Threshold', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

