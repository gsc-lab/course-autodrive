"""학습기 명령.

    uv run -m trainer train dataset_20261110_1906.zip            # 학습 → 시험 분석 → ONNX 내보내기
    uv run -m trainer train a.zip b.zip --name lr0.01 --lr 0.01  # 여러 zip, 실행 이름, 설정 덮어쓰기
    uv run -m trainer test models/<실행 이름>                    # 시험 분석만 다시
    uv run -m trainer test models/<실행 이름> --data other.zip   # 다른 데이터셋 전체로 시험
    uv run -m trainer export models/<실행 이름>                  # ONNX 내보내기만 다시

학습 설정은 config.yaml 의 training 에서 바꾼다.
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="trainer", description="조향 모델 학습 · 시험 분석 · ONNX 내보내기")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="데이터셋 zip 으로 학습한다 (끝나면 시험 분석과 ONNX 내보내기까지)")
    t.add_argument("datasets", nargs="+", help="가공 프로그램이 만든 dataset_*.zip (여러 개 가능)")
    t.add_argument("--config", help="config.yaml 경로 (기본: 저장소 루트)")
    t.add_argument("--name", help="결과 폴더 이름 뒤에 붙일 말 (예: lr0.01)")
    t.add_argument("--epochs", type=int, help="training.epochs 를 이번만 바꾼다")
    t.add_argument("--lr", type=float, help="training.optimizer.lr 을 이번만 바꾼다")
    t.add_argument("--batch-size", type=int, help="training.loader.batch_size 를 이번만 바꾼다")
    t.add_argument("--model", help="training.model 을 이번만 바꾼다")
    t.add_argument("--device", choices=["auto", "cuda", "cpu"], help="training.device 를 이번만 바꾼다")
    t.add_argument("--no-test", action="store_true", help="시험 분석을 하지 않는다")
    t.add_argument("--no-export", action="store_true", help="ONNX 로 내보내지 않는다")

    e = sub.add_parser("test", help="학습된 모델을 시험 데이터로 분석한다")
    e.add_argument("run_dir", help="models/<실행 이름>")
    e.add_argument("--data", nargs="+", help="이 데이터셋 zip 전체를 시험 데이터로 쓴다 (기본: 학습 때 떼어 둔 test)")
    e.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")

    x = sub.add_parser("export", help="학습된 모델을 ONNX 로 내보낸다")
    x.add_argument("run_dir", help="models/<실행 이름>")

    args = p.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):         # 한글을 못 찍는 터미널(영문 Windows 등)에서도 죽지 않게
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    from common.config import ConfigError, load_config, validate_training
    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch 가 설치되어 있지 않습니다. 학습용 패키지를 먼저 설치하세요:\n"
              "  NVIDIA GPU 가 있는 PC :  uv sync --extra gui --extra train\n"
              "  GPU 가 없는 PC        :  uv sync --extra gui --extra train-cpu", file=sys.stderr)
        return 2
    from trainer.data import DatasetError
    from trainer.evaluate import analyze
    from trainer.export import ExportError, export
    from trainer.run import TrainError, train

    try:
        if args.command == "train":
            cfg = load_config(args.config, program="trainer")
            tr = cfg.training
            if args.epochs is not None:
                tr.epochs = args.epochs
            if args.lr is not None:
                tr.optimizer.lr = args.lr
            if args.batch_size is not None:
                tr.loader.batch_size = args.batch_size
            if args.model:
                tr.model = args.model
            if args.device:
                tr.device = args.device
            validate_training(tr)
            run_dir, samples = train(args.datasets, cfg, name=args.name)
            if not args.no_test:
                print("\n[시험 분석] 학습에 쓰지 않은 test 사진으로 잰다")
                analyze(run_dir, samples=samples, device=tr.device)
            if not args.no_export:
                print("\n[모델 배포] ONNX 내보내기")
                export(run_dir)
            print(f"\n끝. 결과 폴더: {run_dir}")
        elif args.command == "test":
            print("[시험 분석]")
            analyze(args.run_dir, data=args.data, device=args.device)
        elif args.command == "export":
            print("[모델 배포] ONNX 내보내기")
            export(args.run_dir)
    except (ConfigError, DatasetError, TrainError, ExportError, ValueError, OSError) as err:
        sys.stdout.flush()                          # 파일로 받을 때도 오류가 진행 기록 뒤에 오게
        print(f"\n오류: {err}", file=sys.stderr)
        return 1
    except torch.OutOfMemoryError:
        sys.stdout.flush()
        print("\n오류: GPU 메모리가 부족합니다. training.loader.batch_size 를 줄이거나, GPU 를 쓰는 다른 프로그램을 닫거나, "
              "--device cpu 로 다시 해 보세요.", file=sys.stderr)
        return 1
    except MemoryError:
        sys.stdout.flush()
        print("\n오류: 메모리가 부족합니다. 데이터셋을 나눠 학습하거나 다른 프로그램을 닫고 다시 해 보세요.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        sys.stdout.flush()
        print("\n중단했습니다.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
