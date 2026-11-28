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


PROG = "uv run -m trainer"          # 도움말의 사용법 줄을 그대로 따라 칠 수 있게, 실제로 치는 명령을 적는다
DEVICES = ["auto", "cuda", "cpu"]

# 도움말의 글은 80칸 터미널에 맞춰 직접 줄을 나눈다 (한글은 두 칸). tests/test_trainer.py 가 폭을 검사한다.
ABOUT = """\
조향 모델을 학습하고, 시험 사진으로 분석하고, 차에서 쓸 ONNX 파일로 내보낸다.

보통은 train 하나면 된다. 학습이 끝나면 시험 분석과 ONNX 내보내기까지 한다.
test 와 export 는 그 두 단계 가운데 하나만 다시 할 때 쓴다.

결과는 models/<실행 이름>/ 폴더 하나에 모인다.
학습 설정(나누는 비율, 에폭, 학습률 등)은 config.yaml 의 training 에서 바꾼다.

명령:
  train    데이터셋 zip 으로 학습한다 (끝나면 시험 분석과 ONNX 내보내기까지)
  test     학습된 모델의 시험 분석을 다시 한다 (다른 데이터셋으로도 할 수 있다)
  export   학습된 모델의 ONNX 내보내기를 다시 한다"""

EXAMPLES = """\
예:
  uv run -m trainer train dataset/dataset_20261123_1751.zip
  uv run -m trainer train dataset/dataset_20261123_1751.zip --lr 0.01
  uv run -m trainer test models/<실행 이름>
  uv run -m trainer test models/<실행 이름> --data dataset/다른날.zip
  uv run -m trainer export models/<실행 이름>

명령마다의 설명과 옵션:
  uv run -m trainer train -h"""

ABOUT_TRAIN = """\
데이터셋 zip 으로 조향 모델을 학습한다. 학습이 끝나면 이어서
  1) 떼어 둔 test 사진으로 시험 분석을 하고
  2) 차에서 쓸 ONNX 파일을 만든다.

결과는 models/<실행 이름>/ 폴더 하나에 모인다 (실행 이름 = 시작한 날짜_시각).
학습 설정은 config.yaml 의 training 에서 바꾼다."""

EXAMPLES_TRAIN = """\
예:
  uv run -m trainer train dataset/dataset_20261123_1751.zip
  uv run -m trainer train dataset/dataset_20261123_1751.zip --lr 0.01
  uv run -m trainer train dataset/A.zip dataset/B.zip --epochs 100 --name AB"""

ABOUT_TEST = """\
학습된 모델을 시험 사진으로 분석한다: 정확도, 기준선, 각도별 재현율, 혼동 행렬.

--data 를 주지 않으면 학습할 때 떼어 둔 test 사진으로 평가한다.
train 이 끝날 때 하는 분석과 같고, 결과는 models/<실행 이름>/ 에 다시 쓴다.

--data 를 주면 그 데이터셋 zip 전체로 평가한다 (다른 날의 주행으로 시험할 때).
결과는 models/<실행 이름>/test_<zip 이름>/ 에 따로 쓴다.
그 zip 에 이 모델을 학습할 때 쓴 사진이 섞여 있으면 몇 장인지 알려 준다."""

EXAMPLES_TEST = """\
예:
  uv run -m trainer test models/20261123_183035
  uv run -m trainer test models/20261123_183035 --data dataset/다른날.zip"""

ABOUT_EXPORT = """\
학습된 모델(model.pth)을 차에서 쓸 ONNX 파일로 내보낸다.
models/<실행 이름>/ 에 model.onnx 와 model_meta.json 을 쓰고,
PyTorch 와 같은 답을 내는지 확인한다.

train 이 끝날 때 자동으로 한다. --no-export 로 건너뛰었거나 다시 만들 때 쓴다.
차로 옮길 때는 models/<실행 이름>/ 폴더를 통째로 옮긴다."""

EXAMPLES_EXPORT = """\
예:
  uv run -m trainer export models/20261123_183035"""


class _Help(argparse.RawTextHelpFormatter):
    """도움말을 적은 줄 그대로 찍는다.

    argparse 는 글자 수로 줄을 접는데 한글은 화면에서 두 칸이라, 맡겨 두면 터미널 폭을 넘겨 줄이 흐트러진다."""

    def add_usage(self, usage, actions, groups, prefix=None):
        super().add_usage(usage, actions, groups, "사용법: " if prefix is None else prefix)


class _Parser(argparse.ArgumentParser):
    """잘못 쓴 명령에는 그 명령의 도움말을 보는 방법을 붙여 준다 (argparse 의 설명 자체는 영어다)."""

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, f"오류: {message}\n도움말: {self.prog} -h\n")


def _help_option(group) -> None:
    group.add_argument("-h", "--help", action="help", help="이 도움말을 보여 주고 끝낸다")


def build_parser() -> argparse.ArgumentParser:
    """명령줄 해석기. 영어로 나오는 argparse 의 제목(usage, options ...)을 쓰지 않으려고 묶음을 직접 만든다."""
    def new(make, *args, **kwargs):
        return make(*args, formatter_class=_Help, add_help=False, **kwargs)

    # 명령 목록은 ABOUT 에 직접 적었다. argparse 가 만드는 목록은 숨긴다: '{train,test,export}' 줄이 하나 더 끼고,
    # 그 줄 때문에 설명이 오른쪽으로 밀려 80칸을 넘는다.
    # 명령을 더하면 ABOUT 의 "명령:" 에도 적는다. tests/test_trainer.py 가 빠진 명령을 잡는다.
    p = new(_Parser, prog=PROG, usage=PROG + " {train,test,export} ...", description=ABOUT, epilog=EXAMPLES)
    sub = p.add_subparsers(dest="command", prog=PROG, help=argparse.SUPPRESS)
    _help_option(p.add_argument_group("옵션"))

    # train ------------------------------------------------------------------
    t = new(sub.add_parser, "train", usage=f"{PROG} train dataset.zip [dataset.zip ...] [옵션]",
            description=ABOUT_TRAIN, epilog=EXAMPLES_TRAIN)
    g = t.add_argument_group("필수")
    g.add_argument("datasets", nargs="+", metavar="dataset.zip",
                   help="가공 프로그램이 만든 dataset_*.zip\n여러 개를 주면 합쳐서 학습한다")
    g = t.add_argument_group("config.yaml 의 값을 이번 실행에서만 바꾸는 옵션 (파일은 그대로다)")
    g.add_argument("--lr", type=float, metavar="LR", help="학습률  training.optimizer.lr")
    g.add_argument("--epochs", type=int, metavar="N", help="에폭 수  training.epochs")
    g.add_argument("--batch-size", type=int, metavar="N", help="배치 크기  training.loader.batch_size")
    g.add_argument("--model", metavar="NAME", help="모델 이름  training.model")
    g.add_argument("--device", choices=DEVICES, metavar="DEVICE", help="계산 장치  training.device  (auto | cuda | cpu)")
    g = t.add_argument_group("그 밖의 옵션")
    g.add_argument("--name", metavar="NAME",
                   help="결과 폴더 이름 뒤에 붙일 말\n예: --name lr0.01 이면 models/20261123_183035_lr0.01")
    g.add_argument("--no-test", action="store_true",
                   help="시험 분석을 지금은 하지 않는다\ntest 사진은 떼어 둔 채로 남아, 나중에 test 명령으로 한다")
    g.add_argument("--no-export", action="store_true",
                   help="ONNX 내보내기를 지금은 하지 않는다\n나중에 export 명령으로 만든다. 그 전에는 차에서 쓸 수 없다")
    g.add_argument("--config", metavar="FILE", help="다른 config.yaml 을 쓴다 (기본: 저장소의 config.yaml)")
    _help_option(g)

    # test -------------------------------------------------------------------
    e = new(sub.add_parser, "test", usage=f"{PROG} test models/<실행 이름> [옵션]",
            description=ABOUT_TEST, epilog=EXAMPLES_TEST)
    g = e.add_argument_group("필수")
    g.add_argument("run_dir", help="학습 결과 폴더  models/<실행 이름>\n(학습이 끝날 때 '결과 폴더:' 뒤에 나온 것)")
    g = e.add_argument_group("옵션")
    g.add_argument("--data", nargs="+", metavar="ZIP",
                   help="이 데이터셋 zip 전체를 시험 사진으로 쓴다\n(주지 않으면 학습할 때 떼어 둔 test 사진)")
    g.add_argument("--device", choices=DEVICES, default="auto", metavar="DEVICE",
                   help="계산 장치  auto | cuda | cpu  (기본: auto)")
    _help_option(g)

    # export -----------------------------------------------------------------
    x = new(sub.add_parser, "export", usage=f"{PROG} export models/<실행 이름>",
            description=ABOUT_EXPORT, epilog=EXAMPLES_EXPORT)
    g = x.add_argument_group("필수")
    g.add_argument("run_dir", help="학습 결과 폴더  models/<실행 이름>")
    _help_option(x.add_argument_group("옵션"))
    return p


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):         # 한글을 못 찍는 터미널(영문 Windows 등)에서도 죽지 않게. 도움말도 한글이다
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    p = build_parser()
    args = p.parse_args(argv)
    if args.command is None:                        # 명령 없이 실행: 영어 오류 한 줄 대신 도움말 전체를 보여 준다
        p.print_help()
        return 2

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
            # 건너뛴 단계는 나중에 하는 명령을 알려 준다: --no-test 는 "지금은 안 한다"이지 test 사진을 없애는 것이 아니다
            if not args.no_test:
                print("\n[시험 분석] 학습에 쓰지 않은 test 사진으로 평가한다")
                analyze(run_dir, samples=samples, device=tr.device)
            elif tr.split.test > 0:
                print(f"\n[시험 분석] 생략 (--no-test). test 사진은 그대로 남아 있다. 나중에 실행:\n"
                      f"      {PROG} test {run_dir}")
            if not args.no_export:
                print("\n[모델 배포] ONNX 내보내기")
                export(run_dir)
            else:
                print(f"\n[모델 배포] 생략 (--no-export). 차에서 쓰려면 먼저 실행:\n"
                      f"      {PROG} export {run_dir}")
            print(f"\n완료. 결과 폴더: {run_dir}")
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
