import ctypes
import numpy as np
import cv2

# Load DLL
lib = ctypes.cdll.LoadLibrary('./watershed-cuda/watershed.dll')

# Define function signature
lib.watershed_run.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_ubyte
]

def run_watershed(img, thresh=127):
    h, w = img.shape
    labels = np.zeros((h, w), dtype=np.int32)

    img_ptr = img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    lbl_ptr = labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    lib.watershed_run(img_ptr, lbl_ptr, w, h, thresh)

    return labels

# Real-time webcam test
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Can't receive frame (stream end?). Exiting ...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray)

    labels = run_watershed(gray, 127)

    # Visualize (map labels to uint8 for display)
    disp = labels.astype(np.uint8)

    cv2.imshow('CUDA Watershed (Dummy)', disp)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
