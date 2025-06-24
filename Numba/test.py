import cv2
import numpy as np
from numba import cuda, uint8, int32

# ---------------------------
# CUDA KERNELS
# ---------------------------

@cuda.jit
def threshold_kernel(gray, mask, thresh):
    x, y = cuda.grid(2)
    if x < gray.shape[0] and y < gray.shape[1]:
        mask[x, y] = uint8(255) if gray[x, y] > thresh else uint8(0)

@cuda.jit
def erode_kernel(mask, out, radius):
    x, y = cuda.grid(2)
    h, w = mask.shape
    if x < h and y < w:
        val = uint8(255)
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                xx = x + dx
                yy = y + dy
                if 0 <= xx < h and 0 <= yy < w:
                    if mask[xx, yy] == 0:
                        val = uint8(0)
                        break
            if val == 0:
                break
        out[x, y] = val

@cuda.jit
def dilate_kernel(mask, out, radius):
    x, y = cuda.grid(2)
    h, w = mask.shape
    if x < h and y < w:
        val = uint8(0)
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                xx = x + dx
                yy = y + dy
                if 0 <= xx < h and 0 <= yy < w:
                    if mask[xx, yy] == 255:
                        val = uint8(255)
                        break
            if val == 255:
                break
        out[x, y] = val

# ---------------------------
# HOST‐SIDE REAL‐TIME LOOP
# ---------------------------

def main(threshold=100, morph_radius=1):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    # First grab one frame to get dimensions
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read from webcam")
    h, w = frame.shape[:2]

    # Allocate GPU arrays
    d_gray    = cuda.device_array((h, w), dtype=np.uint8)
    d_mask    = cuda.device_array((h, w), dtype=np.uint8)
    d_tmp     = cuda.device_array((h, w), dtype=np.uint8)
    d_erode   = cuda.device_array((h, w), dtype=np.uint8)
    d_dilate  = cuda.device_array((h, w), dtype=np.uint8)

    # Choose CUDA block/grid layout
    threadsperblock = (16, 16)
    blockspergrid_x = (h + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (w + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Upload grayscale frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        d_gray.copy_to_device(gray)

        # 1) Threshold
        threshold_kernel[blockspergrid, threadsperblock](d_gray, d_mask, threshold)

        # 2) Opening = Erode → Dilate
        erode_kernel[blockspergrid, threadsperblock](d_mask, d_tmp, morph_radius)
        dilate_kernel[blockspergrid, threadsperblock](d_tmp, d_mask, morph_radius)

        # 3) Closing = Dilate → Erode
        dilate_kernel[blockspergrid, threadsperblock](d_mask, d_tmp, morph_radius)
        erode_kernel[blockspergrid, threadsperblock](d_tmp, d_mask, morph_radius)

        # Download result
        mask = d_mask.copy_to_host()

        # Overlay: keep only foreground pixels
        fg = cv2.bitwise_and(frame, frame, mask=mask)

        # Show side by side
        combined = np.hstack((frame, fg))
        cv2.imshow("Original | Segmented (CUDA kernels)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # You can tweak threshold and radius to suit your lighting/noise conditions
    main(threshold=120, morph_radius=1)
