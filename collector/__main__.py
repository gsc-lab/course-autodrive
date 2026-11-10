from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from common.config import ConfigError, load_config


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="collector", description="차선 유지 RC카 데이터 수집기")
    p.add_argument("--config", help="config.yaml 경로 (기본: 저장소 루트)")
    p.add_argument("--mock", action="store_true", help="PC 테스트: 가짜 차량·카메라, 키보드 입력")
    p.add_argument("--input", choices=["ds4", "keyboard"], help="입력 장치 강제 지정")
    p.add_argument("--skip-checks", action="store_true", help="시작 전 점검 생략 (디버깅용)")
    p.add_argument("--check", action="store_true",
                   help="점검만: 사전 점검 + 카메라·차량·입력 장치 초기화 후 종료 (서보가 중앙으로 움직이면 정상)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"[설정 오류] {e}", file=sys.stderr)
        return 1

    log_dir = Path(cfg.recording.root)
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(log_dir / "collector.log", encoding="utf-8")]
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setLevel(logging.INFO if args.check else logging.WARNING)  # 상태줄과 섞이지 않게 경고 이상만 화면에
    handlers.append(stderr)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers)

    from collector.app import Collector, PreflightError
    from collector.lock import AlreadyRunning, InstanceLock
    lock = InstanceLock(log_dir / "collector.lock")
    try:
        lock.acquire()
    except AlreadyRunning as e:
        print(f"[시작 불가] {e}", file=sys.stderr)
        return 5
    try:
        try:
            app = Collector(cfg, mock=args.mock, input_override=args.input, skip_checks=args.skip_checks,
                            check_only=args.check)
        except PreflightError as e:
            print(f"[시작 불가] {e}", file=sys.stderr)
            return 3
        except Exception as e:
            logging.getLogger("collector").exception("초기화 실패")
            print(f"[초기화 실패] {e}", file=sys.stderr)
            return 4
        if args.check:
            app.close()
            print("[점검 완료] 카메라·차량·입력 장치 초기화 성공. 세션 폴더는 만들지 않았습니다.")
            return 0
        return app.run()
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
