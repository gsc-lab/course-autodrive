# training/RCDataset.py

import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from preprocessor.RCPreprocessor import RCPreprocessor
from preprocessor.RCAugmentor import RCAugmentor


class RCDataset(Dataset):
    """
    RC 자율주행용 Dataset

    - CSV 안에 'split' 컬럼이 있으면 그대로 사용
    - 없으면 servo_angle별 stratified train/test split 수행
    - __getitem__ 에서:
        - 이미지 BGR 로드
        - (train일 때만) 증강
        - RCPreprocessor로 전처리
        - 라벨을 클래스 인덱스로 변환
    """

    def __init__(self,
                 csv_filename,
                 root,
                 preprocessor: RCPreprocessor,
                 augmentor: RCAugmentor = None,
                 split: str = "train",
                 split_ratio: float = 0.8,
                 shuffle: bool = True,
                 random_seed: int = 42):

        self.image_root = root
        self.preprocessor = preprocessor
        self.augmentor = augmentor
        self.split = split

        csv_path = os.path.join(root, csv_filename)
        self.df_full = pd.read_csv(csv_path)

        # ------------------------------------------------------
        # 1) CSV에 split 컬럼이 있으면 그대로 사용
        # ------------------------------------------------------
        if "split" in self.df_full.columns:
            print("[RCDataset] Using existing 'split' column from CSV.")
            self.df = self.df_full[self.df_full["split"] == split].reset_index(drop=True)

        # ------------------------------------------------------
        # 2) 없으면 servo_angle별 stratified split
        # ------------------------------------------------------
        else:
            print("[RCDataset] Performing stratified split...")

            if shuffle:
                self.df_full = self.df_full.sample(frac=1.0, random_state=random_seed)

            df_list = []
            for angle, df_group in self.df_full.groupby("servo_angle"):
                n = len(df_group)
                n_train = int(n * split_ratio)

                if split == "train":
                    df_split = df_group.iloc[:n_train]
                else:
                    df_split = df_group.iloc[n_train:]

                df_list.append(df_split)

            self.df = pd.concat(df_list).reset_index(drop=True)

        # ------------------------------------------------------
        # 3) angle → class index 매핑
        # ------------------------------------------------------
        self.angles = sorted(self.df["servo_angle"].unique().tolist())
        self.angle_to_idx = {a: i for i, a in enumerate(self.angles)}

        print(f"[RCDataset:{split}] samples={len(self.df)}, per-class counts:")
        print(self.df["servo_angle"].value_counts().sort_index())

    # ----------------------------------------------------------
    def __len__(self):
        return len(self.df)

    # ----------------------------------------------------------
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        rel_path = row["image_path"]
        angle = int(row["servo_angle"])
        img_path = os.path.join(self.image_root, rel_path)

        # 1) 이미지 로드 (BGR)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        # 2) train split일 때만 증강
        if self.split == "train" and self.augmentor is not None:
            img_bgr, angle = self.augmentor(img_bgr, angle)

        # 3) 공통 전처리 (추론과 동일)
        img_chw = self.preprocessor(img_bgr)       # np.ndarray, (3, 66, 200), float32
        img_tensor = torch.from_numpy(img_chw).float()

        # 4) 라벨을 클래스 인덱스로 변환
        label = self.angle_to_idx[angle]

        return img_tensor, label


if __name__ == "__main__":
    preproc = RCPreprocessor(out_size=(200, 66),
                             crop_top_ratio=0.4,
                             crop_bottom_ratio=1.0)
    augment = RCAugmentor()

    dataset = RCDataset(
        csv_filename="data_labels_updated.csv",
        root="data-collector/dataset",
        preprocessor=preproc,
        augmentor=augment,
        split="train",
        split_ratio=0.8,
    )

    x, y = dataset[0]
    print("img shape:", x.shape, "label:", y)
