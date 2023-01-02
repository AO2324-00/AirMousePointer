import sys
import os
import numpy as np

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def adjust(img, contrast=1.0, brightness=0.0):
    # 積和演算を行う。
    dst = contrast * img + brightness
    # [0, 255] でクリップし、uint8 型にする。
    return np.clip(dst, 0, 255).astype(np.uint8)