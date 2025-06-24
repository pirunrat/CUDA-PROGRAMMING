import cv2

class CameraStream:
    def __init__(self, cam_id=0):
        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Cannot open camera")

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("❌ Can't receive frame.")
        return frame

    def release(self):
        self.cap.release()

    def show(self, window_name, frame):
        cv2.imshow(window_name, frame)

    def wait_key(self):
        return cv2.waitKey(1)
