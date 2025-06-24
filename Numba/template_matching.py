import cv2
import numpy as np
from numba import cuda

@cuda.jit
def compute_edge_density(gray_img, result, win_h, win_w, stride):
    y, x = cuda.grid(2)
    y = y * stride
    x = x * stride

    if y >= result.shape[0] or x >= result.shape[1]:
        return

    edge_count = 0
    total_pixels = win_h * win_w

    for dy in range(win_h):
        for dx in range(win_w):
            yy = y + dy
            xx = x + dx
            if yy <= 0 or yy >= gray_img.shape[0] - 1 or xx <= 0 or xx >= gray_img.shape[1] - 1:
                continue

            gx = (gray_img[yy, xx + 1] - gray_img[yy, xx - 1])
            gy = (gray_img[yy + 1, xx] - gray_img[yy - 1, xx])
            mag = abs(gx) + abs(gy)

            if mag > 50:
                edge_count += 1

    result[y, x] = edge_count / total_pixels


def detect_objects_cuda(d_gray, d_result, gray_img, win_h, win_w, density_thresh, stride):
    d_gray.copy_to_device(gray_img)

    threads_per_block = (16, 16)
    grid_x = ((d_result.shape[1] + stride - 1) // stride + threads_per_block[0] - 1) // threads_per_block[0]
    grid_y = ((d_result.shape[0] + stride - 1) // stride + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (grid_x, grid_y)

    compute_edge_density[blocks_per_grid, threads_per_block](d_gray, d_result, win_h, win_w, stride)

    result = d_result.copy_to_host()

    # Fast filtering
    ys, xs = np.where(result > density_thresh)
    boxes = [(x, y, win_w, win_h, result[y, x]) for y, x in zip(ys, xs)]
    return boxes


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open webcam")
        return

    win_w, win_h = 32, 32
    density_thresh = 0.2
    stride = 2

    # Read initial frame for setup
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to capture initial frame")
        cap.release()
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h_res = gray.shape[0] - win_h + 1
    w_res = gray.shape[1] - win_w + 1

    # Pre-allocate device memory
    d_gray = cuda.to_device(gray.astype(np.float32))
    d_result = cuda.device_array((h_res, w_res), dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if gray.shape[0] < win_h or gray.shape[1] < win_w:
            continue

        boxes = detect_objects_cuda(d_gray, d_result, gray.astype(np.float32), win_h, win_w, density_thresh, stride)

        for (x, y, w, h, score) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(frame, f"{score:.2f}", (x, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        cv2.imshow("Optimized Real-time Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
