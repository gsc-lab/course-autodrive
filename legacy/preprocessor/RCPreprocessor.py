# preprocessor/RCPreprocessor.py

import cv2
import numpy as np

class RCPreprocessor:
    """
    RC 자율주행용 공통 전처리기 (훈련 + 추론용)
    - 입력: BGR uint8 (H, W, 3)  (OpenCV 기본 포맷)
    - 출력: float32 (3, H, W), [0,1]
    """
    def __init__(self,
                 out_size=(200, 66),      # (width, height)
                 crop_top_ratio=0.4,
                 crop_bottom_ratio=1.0):
        self.out_w, self.out_h = out_size
        self.crop_top_ratio = crop_top_ratio
        self.crop_bottom_ratio = crop_bottom_ratio

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        BGR 이미지를 받아 PilotNet에 들어갈 CHW 텐서 형태로 변환
        """
        h, w, _ = img_bgr.shape

        # 1) 세로 방향 크롭 (위쪽 하늘/보닛 잘라내기)
        y1 = int(h * self.crop_top_ratio)
        y2 = int(h * self.crop_bottom_ratio)
        cropped = img_bgr[y1:y2, :, :]

        # 2) 리사이즈 (width, height)
        resized = cv2.resize(
            cropped,
            (self.out_w, self.out_h),
            interpolation=cv2.INTER_AREA
        )

        # 3) BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 4) [0,255] -> [0,1] float32
        rgb = rgb.astype(np.float32) / 255.0

        # 5) (H, W, C) -> (C, H, W)
        chw = np.transpose(rgb, (2, 0, 1))  # (3, H, W)

        return chw
