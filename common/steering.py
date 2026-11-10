"""조향 각도 모델. 스틱/키 입력을 서보 각도로 바꾸고, discrete 모드에서는 클래스로 양자화한다.

논리 방향: 음수 = 왼쪽, 양수 = 오른쪽. invert=false 이면 왼쪽이 낮은 각도(min 쪽)다.
서보 장착 방향 때문에 반대로 꺾이면 config 의 steering.invert 만 바꾸면 된다.
가공기의 화살표, 학습기의 좌우 플립 증강도 이 모델을 써야 한다.
"""
from __future__ import annotations

from typing import List

from common.config import SteeringConfig


class SteeringModel:
    def __init__(self, cfg: SteeringConfig):
        self.cfg = cfg
        self.mode = cfg.mode
        self.classes: List[int] = sorted(int(c) for c in cfg.classes)
        self.center = int(cfg.center)
        self.min = int(cfg.min)
        self.max = int(cfg.max)
        self.step_deg = int(cfg.step_deg)
        self.invert = bool(cfg.invert)
        self.hysteresis = float(cfg.hysteresis)

    # ------------------------------------------------------------------ 기본
    def clamp(self, angle: float) -> int:
        return int(round(min(self.max, max(self.min, angle))))

    def quantize(self, angle: float) -> int:
        """가장 가까운 클래스. continuous 모드에서도 통계용으로 쓴다."""
        return min(self.classes, key=lambda c: (abs(c - angle), c))

    def class_index(self, angle: float) -> int:
        return self.classes.index(self.quantize(angle))

    def logical_side(self, angle: float) -> float:
        """-1(완전 왼쪽) ~ +1(완전 오른쪽). 화살표 표시용."""
        if angle >= self.center:
            span = self.max - self.center
            v = (angle - self.center) / span if span else 0.0
        else:
            span = self.center - self.min
            v = -(self.center - angle) / span if span else 0.0
        return -v if self.invert else v

    # ------------------------------------------------------------------ 입력 → 각도
    def axis_to_angle(self, x: float) -> float:
        """스틱 값 x ∈ [-1, 1] 을 연속 각도로. 중앙 기준으로 좌우 범위를 각각 선형 매핑."""
        x = max(-1.0, min(1.0, float(x)))
        if self.invert:
            x = -x
        if x >= 0:
            return self.center + x * (self.max - self.center)
        return self.center + x * (self.center - self.min)

    def from_axis(self, x: float, current: int) -> int:
        raw = self.axis_to_angle(x)
        if self.mode == "continuous":
            return self.clamp(raw)
        candidate = self.quantize(raw)
        if candidate == current or current not in self.classes:
            return candidate
        ci, ki = self.classes.index(current), self.classes.index(candidate)
        if abs(ci - ki) > 1:
            return candidate  # 두 칸 이상 떨어지면 바로 전환
        boundary = (current + candidate) / 2.0
        margin = self.hysteresis * abs(candidate - current)
        if candidate > current:
            return candidate if raw >= boundary + margin else current
        return candidate if raw <= boundary - margin else current

    def step(self, current: int, direction: int) -> int:
        """direction: -1 왼쪽, +1 오른쪽 (논리 방향). discrete 는 한 클래스, continuous 는 step_deg."""
        d = -direction if self.invert else direction
        if self.mode == "continuous":
            return self.clamp(current + d * self.step_deg)
        if current not in self.classes:
            current = self.quantize(current)
        i = self.classes.index(current) + (1 if d > 0 else -1)
        i = max(0, min(len(self.classes) - 1, i))
        return self.classes[i]

    def mirror(self, angle: int) -> int:
        """좌우 플립 증강용: 중앙 기준 대칭 각도. discrete 면 클래스로 양자화."""
        mirrored = 2 * self.center - angle
        mirrored = self.clamp(mirrored)
        return self.quantize(mirrored) if self.mode == "discrete" else mirrored


def recommend_steering(low_limit: int, high_limit: int, straight: int | None = None,
                       margin: int = 3, n_classes: int = 5) -> dict:
    """실측한 기구 한계(서보 각도)에서 config.yaml 의 steering 값을 계산한다.

    low_limit / high_limit: 조향 링크가 걸리기 직전의 낮은 쪽·높은 쪽 서보 각도.
    straight: 바퀴가 정면일 때의 각도 (없으면 두 한계의 가운데).
    좌우 플립 증강과 균형 잡힌 조향을 위해 좁은 쪽에 맞춘 대칭 범위를 쓰고, 한계에서 margin 도만큼 안쪽으로 들인다.
    """
    if not (low_limit < high_limit):
        raise ValueError(f"낮은 쪽 한계가 높은 쪽 한계보다 작아야 합니다: {low_limit}, {high_limit}")
    if n_classes < 3 or n_classes % 2 == 0:
        raise ValueError(f"클래스 수는 3 이상의 홀수여야 합니다: {n_classes}")
    center = int(round((low_limit + high_limit) / 2.0)) if straight is None else int(straight)
    half = min(center - low_limit, high_limit - center) - margin
    if half < (n_classes - 1) // 2:
        raise ValueError(f"조향 범위가 너무 좁습니다: 중앙 {center}, 한계 {low_limit}~{high_limit}, 여유 {margin}")
    k = (n_classes - 1) // 2
    classes = [int(round(center + i * half / k)) for i in range(-k, k + 1)]
    return {"center": center, "min": classes[0], "max": classes[-1], "classes": classes}
