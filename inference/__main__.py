"""추론기 명령.

    uv run python -m inference models/<실행 이름>            # 자율주행 (패드의 ○ 로 시작/정지)
    uv run python -m inference models/<실행 이름> --check    # 점검만: 엔진 만들기 + 카메라, 보드, 패드 (서보가 가운데로 한 번 움직인다)
    uv run python -m inference models/<실행 이름> --mock     # PC 에서 흉내만: 가짜 차량·카메라, 키보드, onnxruntime
    uv run python -m inference.engine models/<실행 이름>     # 엔진만 만들어 보고 추론 시간을 잰다 (차는 움직이지 않는다)

모델 폴더는 PC 의 학습 결과 models/<실행 이름>/ 을 통째로 차에 복사한 것이다 (model.onnx 와 model_meta.json 이 있어야 한다).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from common.config import ConfigError, load_config


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="inference", description="학습한 모델로 자율주행 (패드의 ○ 로 시작/정지)")
    p.add_argument("model", help="모델 폴더: models/<실행 이름>")
    p.add_argument("--config", help="config.yaml 경로 (기본: 저장소 루트)")
    p.add_argument("--mock", action="store_true", help="PC 테스트: 모의 차량·카메라, 키보드 입력, onnxruntime")
    p.add_argument("--input", choices=["ds4", "keyboard"], help="입력 장치 강제 지정")
    p.add_argument("--backend", choices=["auto", "tensorrt", "onnxruntime"], default="auto",
                   help="모델을 실행할 도구 (기본 auto: Jetson 은 tensorrt, PC 는 onnxruntime)")
    p.add_argument("--skip-checks", action="store_true", help="시작 전 점검 생략 (디버깅용)")
    p.add_argument("--check", action="store_true",
                   help="점검만: 사전 점검 + 엔진 만들기 + 카메라·차량·입력 장치 초기화 후 종료 (서보가 중앙으로 움직이면 정상)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    try:
        cfg = load_config(args.config, program="inference")    # 차가 읽는 절만 검사: 학습 값의 오타가 차를 세우지 않게
    except ConfigError as e:
        print(f"[설정 오류] {e}", file=sys.stderr)
        return 1

    from inference.bundle import BundleError, load_bundle
    try:
        bundle = load_bundle(args.model)                        # 장치를 열기 전에: 모델 폴더가 잘못됐으면 바로 알린다
    except BundleError as e:
        print(f"[모델 오류] {e}", file=sys.stderr)
        return 6

    from common.bgio import BackgroundFileHandler              # 로그 파일은 뒤의 스레드가 쓴다 (주 루프가 SD 카드에 막히지 않게)
    from common.driveapp import DroppingStderrHandler, PreflightError
    log_dir = Path(cfg.recording.root)
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    screen = DroppingStderrHandler(level=logging.INFO if args.check else logging.WARNING)   # 끊긴 터미널에서 멈추지 않는다
    screen.setFormatter(logging.Formatter(fmt))
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=fmt,
                        handlers=[BackgroundFileHandler(log_dir / "inference.log"), screen])

    from common.lock import AlreadyRunning, InstanceLock
    from inference.app import Autopilot
    from inference.engine import EngineError
    from inference.predict import PredictError
    lock = InstanceLock(log_dir / "collector.lock")             # 수집기와 같은 잠금: 둘이 함께 돌면 카메라와 I2C 를 다툰다
    try:
        lock.acquire()
    except AlreadyRunning as e:
        print(f"[시작 불가] {e}", file=sys.stderr)
        return 5
    try:
        try:
            app = Autopilot(cfg, bundle, mock=args.mock, input_override=args.input, skip_checks=args.skip_checks,
                            check_only=args.check, backend=args.backend)
        except PreflightError as e:
            print(f"[시작 불가] {e}", file=sys.stderr)
            return 3
        except (BundleError, EngineError, PredictError) as e:
            print(f"[모델 오류] {e}", file=sys.stderr)
            return 6
        except KeyboardInterrupt:                               # 준비하는 동안 Ctrl+C: 연 장치는 이미 놓았다
            print("\n[중단] 시작하기 전에 멈췄습니다.", file=sys.stderr)
            return 130
        except Exception as e:
            logging.getLogger("inference").exception("초기화 실패")
            print(f"[초기화 실패] {e}", file=sys.stderr)
            return 4
        if args.check:
            report, blocked = app.check_report(), app.logic.block_reason
            app.close()
            if blocked:                                         # 모델과 장치는 됐지만, 이대로는 ○ 를 눌러도 달리지 않는다
                print(f"[점검 실패] {report}\n자율주행을 켤 수 없습니다: {blocked}", file=sys.stderr)
                return 7
            print(f"[점검 완료] {report}")
            return 0
        return app.run()
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
