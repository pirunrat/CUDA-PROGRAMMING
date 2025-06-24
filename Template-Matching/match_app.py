import cv2
from pyramid_match import TemplateMatcher
from video_capture import CameraStream
import numpy as np

class MatchApp:
    def __init__(self, matcher, template_path, scales, ncc_thresh=0.7):
        self.matcher = matcher
        self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if self.template is None:
            raise RuntimeError("❌ Template not found!")
        self.scales = scales
        self.ncc_thresh = ncc_thresh
        self.tmpl_mean, self.tmpl_var = self.matcher.compute_template_stats(self.template)

    def adjust_template_size(self, frame):
        frame_h, frame_w = frame.shape[:2]
        min_scale = min(self.scales)
        max_tmpl_h = int(frame_h * min_scale)
        max_tmpl_w = int(frame_w * min_scale)
        if self.template.shape[0] > max_tmpl_h or self.template.shape[1] > max_tmpl_w:
            scale_factor = min(max_tmpl_h / self.template.shape[0], max_tmpl_w / self.template.shape[1])
            self.template = cv2.resize(self.template, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            print(f"✅ Resized template to {self.template.shape}")
            self.tmpl_mean, self.tmpl_var = self.matcher.compute_template_stats(self.template)

    def run(self):
        cam = CameraStream()
        frame = cam.read_frame()
        self.adjust_template_size(frame)

        while True:
            frame = cam.read_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            best_val = -np.inf
            best_loc = None
            best_scale = 1.0

            for scale in self.scales:
                scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                if scaled.shape[0] < self.template.shape[0] or scaled.shape[1] < self.template.shape[1]:
                    continue

                result = self.matcher.match(scaled, self.template, self.tmpl_mean, self.tmpl_var)
                max_val = result.max()
                max_loc = np.unravel_index(result.argmax(), result.shape)

                if max_val > best_val:
                    best_val = max_val
                    best_loc = (int(max_loc[1] / scale), int(max_loc[0] / scale))
                    best_scale = scale

            if best_loc and best_val > self.ncc_thresh:
                top_left = best_loc
                br = (top_left[0] + int(self.template.shape[1] / best_scale),
                      top_left[1] + int(self.template.shape[0] / best_scale))
                cv2.rectangle(frame, top_left, br, (0, 255, 0), 2)
                print(f"✅ Match at scale {best_scale:.2f} with NCC {best_val:.4f}")
            else:
                print(f"❌ No good match this frame (NCC {best_val:.4f})")

            cam.show('CUDA NCC Pyramid Match', frame)

            if cam.wait_key() == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    matcher = TemplateMatcher('./template-matching-cuda/pyramid_match.dll')
    app = MatchApp(matcher, 'me.jpg', scales=[1.0, 0.75, 0.5], ncc_thresh=0.7)
    app.run()
