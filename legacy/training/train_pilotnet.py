# training/train_pilotnet.py

import os
import time
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from training.RCDataset import RCDataset
from preprocessor.RCPreprocessor import RCPreprocessor
from preprocessor.RCAugmentor import RCAugmentor
from training.model import PilotNet

# 고정 입력 크기에서 cuDNN 최적화
torch.backends.cudnn.benchmark = True


def train():
    # 학습 설정
    csv_filename = "data_labels_updated.csv"
    dataset_root = "datacollector/dataset"
    num_epochs = 20
    batch_size = 128
    learning_rate = 5e-4
    weight_decay = 1e-4
    split_ratio = 0.8

    # 학습 장치 선택
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # PilotNet 입력 크기로 전처리
    preproc = RCPreprocessor(
        # PilotNet 기본 입력: width=200, height=66
        out_size=(200, 66),
        # 상단 배경을 일부 제거
        crop_top_ratio=0.4,
        crop_bottom_ratio=1.0
    )

    # 데이터 증강 옵션(현재 미사용)
    augment = RCAugmentor(
        hflip_prob=0.5,
        brightness_delta=0.2,
        blur_prob=0.3
    )

    # train/test 데이터셋 구성
    train_dataset = RCDataset(
        csv_filename=csv_filename,
        root=dataset_root,
        preprocessor=preproc,
        # 증강을 쓰려면 None 대신 augment 전달
        augmentor=None,
        split="train",
        split_ratio=split_ratio
    )

    test_dataset = RCDataset(
        csv_filename=csv_filename,
        root=dataset_root,
        preprocessor=preproc,
        # 검증 데이터는 증강하지 않음
        augmentor=None,
        split="test",
        split_ratio=split_ratio
    )

    # 조향 각도 클래스를 출력 클래스 수로 사용
    num_classes = len(train_dataset.angles)
    print(f"[INFO] classes = {num_classes}")
    print(f"[INFO] train samples = {len(train_dataset)}")
    print(f"[INFO] test  samples = {len(test_dataset)}")

    # DataLoader 성능 옵션
    pin_memory = (device.type == "cuda")
    num_workers = 12 if device.type == "cuda" else 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # 학습 데이터는 매 epoch 섞음
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # worker를 유지해 반복 생성 비용 감소
        persistent_workers=True,
        # 미리 가져올 배치 수
        prefetch_factor=4,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        # 검증 데이터는 순서 유지
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # 모델, 손실 함수, 옵티마이저
    # 입력 텐서 형태: (채널, 높이, 너비)
    model = PilotNet(num_classes=num_classes, input_shape=(3, 66, 200)).to(device)

    # label smoothing으로 overconfidence 완화
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)

    # 학습 및 검증
    train_start = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()

        # epoch별 누적 지표 초기화
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # 데이터 이동/연산 시간 분리 측정
        epoch_start = time.time()
        data_move_time = 0.0
        compute_time = 0.0

        for images, labels in train_loader:
            t0 = time.time()
            # pin_memory와 함께 GPU 전송 비동기화
            images = images.to(device, non_blocking=True)
            labels = labels.to(device)
            t1 = time.time()
            data_move_time += (t1 - t0)

            # 이전 배치의 gradient 제거
            optimizer.zero_grad()

            t2 = time.time()
            # logits 계산 후 손실 역전파
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            t3 = time.time()
            compute_time += (t3 - t2)

            # 배치 평균 손실을 샘플 수 기준으로 누적
            train_loss += loss.item() * images.size(0)

            # 가장 큰 logit을 예측 클래스로 선택
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total * 100.0

        # 검증
        model.eval()

        # 검증 지표 초기화
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        # 검증 중 gradient 계산 비활성화
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                # 검증 손실도 샘플 수 기준으로 누적
                test_loss += loss.item() * images.size(0)

                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        epoch_test_loss = test_loss / test_total
        epoch_test_acc = test_correct / test_total * 100.0

        epoch_time = time.time() - epoch_start

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={epoch_train_loss:.4f}, train_acc={epoch_train_acc:.2f}% | "
            f"test_loss={epoch_test_loss:.4f}, test_acc={epoch_test_acc:.2f}% | "
            f"time={epoch_time:.2f}s "
            f"(data={data_move_time:.2f}s, compute={compute_time:.2f}s)"
        )

    print(f"Total train time={time.time()-train_start:.2f}s")

    # 모델 저장 경로
    os.makedirs("models", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # PyTorch 가중치 저장
    pth_path = f"models/pilotnet_steering_{timestamp}.pth"
    torch.save(model.state_dict(), pth_path)
    print(f"[INFO] Saved PTH → {pth_path}")

    # ONNX 내보내기
    onnx_path = f"models/pilotnet_steering_{timestamp}.onnx"
    # ONNX export용 예시 입력
    dummy_input = torch.randn(1, 3, 66, 200, dtype=torch.float32).to(device)

    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=11,
        # 학습된 파라미터 포함
        export_params=True,
        # 상수 연산 최적화
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        # 고정 입력 크기 사용
        dynamic_axes=None,
    )

    print(f"[INFO] Saved ONNX → {onnx_path}")


if __name__ == "__main__":
    train()
