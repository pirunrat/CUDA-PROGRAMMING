import ctypes
import numpy as np

class TemplateMatcher:
    def __init__(self, lib_path):
        self.lib = ctypes.cdll.LoadLibrary(lib_path)
        self.lib.match_template.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float,
            ctypes.POINTER(ctypes.c_float)
        ]

    def compute_template_stats(self, template):
        mean = template.mean()
        var = ((template.astype(np.float32) - mean) ** 2).mean()
        return float(mean), float(var)

    def match(self, img, tmpl, tmpl_mean, tmpl_var):
        img_h, img_w = img.shape
        tmpl_h, tmpl_w = tmpl.shape

        result = np.zeros((img_h, img_w), dtype=np.float32)

        img_ptr = img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        tmpl_ptr = tmpl.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        res_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self.lib.match_template(img_ptr, img_w, img_h,
                                tmpl_ptr, tmpl_w, tmpl_h,
                                tmpl_mean, tmpl_var,
                                res_ptr)
        return result
