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

torch.backends.cudnn.benchmark = True


def train():
    # =====================
    # 1. Hyperparameters
    # =====================
    csv_filename = "data_labels_updated.csv"
    dataset_root = "datacollector/dataset"
    num_epochs = 20
    batch_size = 128
    learning_rate = 5e-4
    weight_decay = 1e-4
    split_ratio = 0.8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # =====================
    # 2. Dataset & Loader
    # =====================
    preproc = RCPreprocessor(
        out_size=(200, 66),
        crop_top_ratio=0.4,
        crop_bottom_ratio=1.0
    )

    # Train에만 augmentation 적용/ 사용여부 검토
    augment = RCAugmentor(
        hflip_prob=0.5,
        brightness_delta=0.2,
        blur_prob=0.3
    )

    train_dataset = RCDataset(
        csv_filename=csv_filename,
        root=dataset_root,
        preprocessor=preproc,
        augmentor=None,
        split="train",
        split_ratio=split_ratio
    )

    test_dataset = RCDataset(
        csv_filename=csv_filename,
        root=dataset_root,
        preprocessor=preproc,
        augmentor=None,
        split="test",
        split_ratio=split_ratio
    )

    num_classes = len(train_dataset.angles)
    print(f"[INFO] classes = {num_classes}")
    print(f"[INFO] train samples = {len(train_dataset)}")
    print(f"[INFO] test  samples = {len(test_dataset)}")

    pin_memory = (device.type == "cuda")
    num_workers = 12 if device.type == "cuda" else 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=4,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # =====================
    # 3. Model / Loss / Optim
    # =====================
    model = PilotNet(num_classes=num_classes, input_shape=(3, 66, 200)).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)

    # =====================
    # 4. Train + Eval Loop
    # =====================
    train_start = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        epoch_start = time.time()
        data_move_time = 0.0
        compute_time = 0.0

        for images, labels in train_loader:
            t0 = time.time()
            images = images.to(device, non_blocking=True)
            labels = labels.to(device)
            t1 = time.time()
            data_move_time += (t1 - t0)

            optimizer.zero_grad()

            t2 = time.time()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            t3 = time.time()
            compute_time += (t3 - t2)

            train_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total * 100.0

        # ===== Eval =====
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

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

    # =====================
    # 5. Save Model + ONNX
    # =====================
    os.makedirs("models", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # PyTorch 저장
    pth_path = f"models/pilotnet_steering_{timestamp}.pth"
    torch.save(model.state_dict(), pth_path)
    print(f"[INFO] Saved PTH → {pth_path}")

    # ONNX 저장
    onnx_path = f"models/pilotnet_steering_{timestamp}.onnx"
    dummy_input = torch.randn(1, 3, 66, 200, dtype=torch.float32).to(device)

    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=11,
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )

    print(f"[INFO] Saved ONNX → {onnx_path}")


if __name__ == "__main__":
    train()
