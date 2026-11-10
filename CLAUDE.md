# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Capstone project (영진전문대 글로벌시스템융합과): a mini autonomous RC car that does lane keeping with a
small CNN. Code and comments are in Korean. Conversation with the user is in Korean.

## Status: full rewrite in progress (decided 2026-11-04)

The hardware moved from Jetson Nano to **Jetson Orin Nano / Super** (JetPack 6.x, TensorRT 10, Python 3.10)
and the software is being rewritten from scratch with new features. **Everything under `legacy/` is the
old Jetson Nano code.** Since 2026-11-10 it is **untracked and gitignored** (user decision: not worth committing);
it exists only as a local reference on the professor's PC, and the old code remains in git history before
commit `d2e0de0`. Do not extend it, do not reuse its hardware code, and do not treat its conventions as binding.

Hardware bring-up (2026-11-09): servo (PCA9685 0x40 CH0) and both motors (PCA9685 0x41 CH0/CH1 → MDD10A) were
verified on the reference car with `scripts/pin_test.py servo` / `motor 15`, and a first real `collector` drive with the
DualShock 4 worked. Not yet done: confirming `dir_forward_high` / `steering.invert` on the track, and a hardware stop for
I2C loss (PCA9685 has no watchdog; the car has no main switch, so pulling the battery connector is the only stop).

Current repo state (2026-11-10): `config.yaml`, `common/`, `collector/` and `tests/` exist and pass
`pytest`; `curator/`, `trainer/`, `inference/` are not written yet. `docs/jetson-setup.md` holds the
one-time Jetson setup (packages, i2c/input permissions, DualShock 4 pairing, I2C check). **The header-PWM
(`jetson_pwm`, Jetson.GPIO, jetson-io pinmux, pin-29 overlay patch) code path was removed entirely on 2026-11-10**
at the user's request: every actuator signal comes from the two PCA9685 boards over I2C. Do not reintroduce
`hardware.pins`, `*.driver`, `safety.pwm_check`, Jetson.GPIO or any `/sys/class/pwm` check.

## Commands

Dependencies are managed with **uv** (`pyproject.toml` + `uv.lock`, Python pinned to 3.10 in
`.python-version` to match JetPack 6). Do not add `requirements*.txt` files or use `pip install` directly;
add dependencies with `uv add` (core) / `uv add --optional pc|jetson` / `uv add --group dev`.

```bash
uv sync --group dev                    # core + pytest (fast; enough for tests and --mock)
uv sync --extra pc                     # + PySide6, torch 2.11 cu128, onnx (multi-GB download, PC only)
uv run pytest                          # unit tests + a mock end-to-end collector run (no hardware)
uv run pytest tests/test_state.py -k deadman
uv run python -m collector --mock      # PC dry run: mock vehicle + synthetic camera + keyboard (needs a real tty)
uv run python -m collector             # Jetson: DualShock 4; add --input keyboard for SSH keyboard
uv run python -m collector --check     # Jetson: preflight (I2C bus, both PCA9685 boards) + open camera/input, then exit (servo centers once)
uv run python -m collector --skip-checks -v   # bypass preflight (debugging only), debug log to dataset/collector.log
uv run python scripts/pin_test.py scan|servo|motor|servo-all|servo-hold   # Jetson: I2C scan / actuator tests (wheels off the ground)
uv run python scripts/dotmatrix_test.py [status|stress|off]              # Jetson: MAX7219 status display test
bash scripts/jetson_setup.sh install|verify|pair|prepare-image   # Jetson provisioning (idempotent), see docs/jetson-setup.md
```

Student boards are provisioned by **cloning a golden SD image** (professor runs `install` → `--check` →
`prepare-image`, then images the card); students only run `pair` for their own DualShock 4. The install step
sets `ClassicBondedOnly=false` in `/etc/bluetooth/input.conf`: bluez otherwise rejects the DS4's
pre-bonding HID connection. The user decided on 2026-11-04 that this is acceptable because the images are
personal lab images with no sensitive data; keep it, do not "harden" it away.

The reference board is a Jetson Orin Nano Developer Kit Super on **JetPack 6.2.1** (L4T 36.4.7, Python
3.10.12, TensorRT 10.3). JetPack 7.2.x was evaluated on 2026-11-04 and rejected for now (Python 3.12, no SD
image); `docs/jetson-setup.md` has the notes. Provisioning a fresh image must go through
`scripts/jetson_setup.sh` so every board ends up identical; do not hand-document steps that the script
does not perform.

On the Jetson, create the venv first with `uv venv --system-site-packages` so JetPack's tensorrt /
cuda-python are visible, then `uv sync` (see `docs/jetson-setup.md`); evdev is a core dependency with a linux marker, not an extra, because plain `uv run` strips extras. torch comes from the
PyTorch cu128 index on x86-64 only; Jetson never installs torch (no torch marker for aarch64). `tool.uv.environments`
restricts the lock to x86-64 Windows/Linux and aarch64 Linux.

Everything runs from the repo root as a module (`python -m collector`); `pyproject.toml` sets `pythonpath`
for pytest. The collector logs to `dataset/collector.log` and writes sessions to `dataset/<session>/`;
`dataset/` and `models/` are gitignored. The hardware and input drivers under `common/hw/` and
`common/inputs/` are lazy-importing, so `common` imports cleanly on Windows. `collector/state.py`
(`DriveLogic`) holds all drive/record rules with no I/O and is where behaviour changes should go first,
with a test in `tests/test_state.py`.

**The current wiring is `docs/wiring.pdf` (R5.1, 2026-11-10): 31 conductors, generated by `scripts/make_wiring_doc.py`
from one netlist (`NET`) so the figures and tables cannot disagree; the script lints text/line overlaps and
`tests/test_wiring_doc.py` guards it. Change wiring there, not by hand-editing the HTML.** R5.1 has 7 ground conductors:
the Jetson's only ground is header pin 6 → servo board GND (in the I2C cable), and the MDD10A logic ground comes from the
motor board's channel-0 GND pin, not from the star ground. Motor signals use only CH0 (PWM) and CH1 (DIR) of the motor
board and are split to the two MDD10A channels by two WAGO 221 3-port blocks: J4 (CH0 → PWM1 + PWM2) and J5 (CH1 → DIR1 +
DIR2), 22 AWG wire only (Dupont jumpers are below WAGO's 24 AWG minimum). Decided 2026-11-10: "both motors get the same
command" is guaranteed by wiring, not by software; `pwm_channel_2`/`dir_channel_2` default to −1 and exist only for boards
wired with four separate channels. **There is no main power switch on the car** (user, 2026-11-10); the only stop on
comms loss is pulling the drive-battery connector. Older wiring revisions (R3 PDF/PNG, R4 HTML) were deleted on
2026-11-10; `docs/wiring.pdf` ("자율주행자동차 배선도 Version 1.0") is the only wiring document. Connection map (40-pin
header J12, BOARD numbering):

| Function | Where | Notes |
|---|---|---|
| Motor PWM → J4 → MDD10A PWM1 + PWM2 | PCA9685 #2 (0x41) CH0 | `hardware.motor.pwm_hz` 1500 (PCA9685 max 1526) |
| Motor DIR → J5 → MDD10A DIR1 + DIR2 | PCA9685 #2 CH1 (full on/off) | left motor on M1A/M1B, right on M2A/M2B. Optional CH2/CH3 mirroring (`pwm_channel_2`, `dir_channel_2`, default −1) for boards wired with four separate channels |
| Servo signal (LD-1501MG) | PCA9685 #1 (0x40) CH0 | 50 Hz, 500–2500 µs, neutral 1500 µs |
| I2C to both boards (chained) | header pins 3 (SDA), 5 (SCL), 1 (3.3 V VCC), 6 (GND) | bus i2c-7; servo power = UBEC 6 V on board #1's V+ terminal block; these four are the only header pins used |

**All actuators go through PCA9685 boards over I2C (decided 2026-11-06, header path deleted 2026-11-10).** The Orin Nano
devkit header drives its 40-pin outputs through level shifters: with the LD-1501MG connected, pin 15 could only pull the
signal line down to about 1.4 V (DC average 1.5 V instead of 0.25 V), so the servo never saw a valid pulse, while the same
servo moved immediately from a PCA9685 (`common/hw/pca9685.py`, dependency-free `/dev/i2c` access). One PCA9685 has a
single PWM frequency for all 16 channels, so the servo (50 Hz) and the motor PWM (1.5 kHz) need two boards; board #2 has
its A0 pad bridged (0x41) and is chained on the same SDA/SCL/VCC/GND. The earlier "servo never moves" hunt was first
caused by missing servo V+ power (reversed terminal-block wires); check V+ at the servo header pins with a meter before
suspecting software. `scripts/pin_test.py scan` lists the I2C addresses that answer and says which board is missing;
`collector --check` runs `check_i2c_bus` and `check_pca9685` for both boards (`common/hw/checks.py`).

Hard rules that come from the hardware:
- **Actuators are I2C only.** Programs must verify `/dev/i2c-7` access and both PCA9685 boards (0x40, 0x41) at startup
  and refuse to run with a clear message naming the missing board. Never drive the header's PWM/GPIO pins, never use
  Jetson.GPIO or `busybox devmem`. PCA9685 keeps its last output when I2C drops, so stop logic must always write duty 0 last.
- **The only jetson-io use is SPI1 for the optional MAX7219 status display** (`common/display/`, header pins 19 MOSI /
  23 SCK / 24 CS0 plus 2 = 5 V and 20 = GND, `/dev/spidev0.0`, user in group `gpio`). It is wired directly with 3.3 V
  logic although the MAX7219 datasheet asks for 3.5 V: tested 2026-11-10 with 10 minutes of continuous pattern
  changes without a glitch, so no level shifter. The display is optional: `create_display` returns `NullDisplay`
  when `display.enabled` is false, the device is missing or the open fails, and the collector keeps driving.
  `scripts/jetson_setup.sh install` enables spi1 (reboot needed); `scripts/dotmatrix_test.py status` previews the screens.
- Motor driver is Cytron MDD10A: one DIR line + PWM duty per channel (no IN1/IN2 pairs). Both channels receive the
  same PWM and DIR through WAGO splitters J4/J5, so the two rear motors always receive the same command.
- Servo is powered by a 6 V UBEC, not by the Jetson. The Jetson has its own USB-PD power bank. The PCA9685 terminal
  block is reverse-polarity protected, so reversed UBEC wires show up only as "V+ pin reads 0 V".
- The car has **no monitor**. Nothing on the Jetson may open an OpenCV window.

## Target architecture

```
common/      shared by every program: config loading, filename format/parse, steering class
             quantization, image preprocessing, hardware drivers (Jetson-only, lazy-imported),
             input devices (DualShock 4 via evdev, keyboard via terminal raw mode),
             display/ (optional MAX7219 8×8 status display over SPI; glyphs.py is pure and tested)
collector/   Jetson. Drives the car with a DualShock 4, records frames to dataset/<session>/
curator/     PC. PySide6 GUI to review sessions, exclude frames, build a dataset zip
trainer/     PC. Trains from dataset zips, exports ONNX + model_meta.json
inference/   Jetson. Builds a TensorRT 10 engine from ONNX on-device and drives the car
config.yaml  single source of truth for steering classes, PCA9685 addresses/channels, camera, recording, training
```

Single-source rules: the steering class list, the filename format, and the preprocessing parameters each
live in exactly one place (`config.yaml` + `common/`) and every program imports them. The legacy code had
the class list copied into four files; do not recreate that.

Dependencies are split into uv extras: `pc` (PySide6, torch, onnx, pandas) and `jetson` (evdev; tensorrt and
cuda-python come from JetPack through `--system-site-packages`). Jetson-only modules must be importable on the PC
without side effects at import time (no device opens, no terminal mode changes at import).

## Contracts between stages

**Collector → curator: session folder** `dataset/YYYYMMDD_HHMM/`
- Frames `YYYYMMDD_HHMMSS_mmm_a{angle:03d}_s{speed:03d}.jpg`, 640×480, JPEG q95. The filename is the only
  label. Angle is the real servo angle in integer degrees in both steering modes.
- Only frames recorded while driving forward with speed > 0 are saved.
- `session_meta.json` (config snapshot, camera settings, per-class counts), `events.csv` (every input
  event with timestamp), `marks.csv` (driver-flagged bad intervals, □ button).

**Curator → trainer: dataset zip** `dataset_YYYYMMDD_HHMM.zip`
- `images/YYYYMMDD_HHMM_{index:06d}.jpg` renumbered in capture order; originals are never modified.
- `labels.csv`, UTF-8, English header: `index,image,servo_angle,motor_speed,source` (`source` = original
  filename, for traceability).
- `meta.json`: `schema_version`, `steering` (mode, classes, min, max), `image` (width, height, format),
  `sessions` (list of source session folders), per-class counts, number excluded, tool version.
- Curation decisions are soft: `curation.json` beside each session records excluded frames; nothing is
  deleted on disk. Multiple sessions can be merged into one zip.

**Trainer → inference: model bundle** `models/<run_id>/`
- `model.pth`, `model.onnx` (fixed batch 1, ONNX input name `input`, output name `output`).
- `model_meta.json`: `task` (`classification` | `regression`), `classes` (for classification), the exact
  preprocessing parameters (crop ratios, output size, color order RGB, scale 0–1, layout CHW), input
  shape, output semantics (logits over `classes`, or normalized angle and the formula back to degrees),
  training run info. **Inference reads preprocessing from `model_meta.json`, not from `config.yaml`**, so
  a model can never be run with the wrong preprocessing.

**Inference (Jetson)**
- Builds and caches `model.engine` from `model.onnx` on the device (engines are not portable across
  boards or TensorRT versions); cache key = ONNX hash + TensorRT version.
- Uses the TensorRT 10 API (`get_tensor_name`, `set_tensor_address`, `execute_async_v3`). The legacy
  `get_binding_index` / `execute_async_v2` API is removed in TensorRT 10.
- Reuses `common/` hardware drivers and the DualShock 4 for manual override; autopilot still needs a
  hold-to-run or stop control of its own (the collector no longer uses L1; decide when inference is written). Prediction smoothing (EMA or majority vote) is configurable. Logs predictions to CSV.

## Collector behaviour (B1, confirmed 2026-11-04)

- Steering `mode: discrete` (default) snaps the left stick to the nearest class in `steering.classes`
  (default `[30, 60, 90, 120, 150]`, students may edit) with hysteresis; `mode: continuous` uses
  `step_deg`. D-pad left/right steps one class.
- DualShock 4 (CUH-ZCT2G, found by USB id 054c:09cc, ignoring the touchpad and motion-sensor devices):
  left stick X steering, R2 forward (trigger depth is quantized into `throttle.levels`, default `[30, 45, 60]` %,
  with a deadzone and hysteresis, so the speed label in filenames only takes those values; releasing it stops; an
  empty list means continuous 0..`throttle.max_speed`), L2 reverse (fixed `reverse_max_speed`, never recorded), ○ record
  toggle, □ mark last 3 s bad, △ center, Options quit. **The JetPack 6.2.1 kernel has no hid-sony /
  hid-playstation driver**, so the pad attaches via hid-generic with different evdev codes and no light bar
  or rumble. `common/inputs/ds4.py` detects the driver from sysfs and selects the `generic` or `sony` layout
  (`LAYOUTS`); `config.yaml` `input.ds4.layout` / `codes` override it. `uv run python -m common.inputs.ds4`
  prints raw and translated events on the board to verify a mapping. Status is shown on the terminal status
  line; the light bar/rumble only work under a sony-class driver. Keyboard over SSH is the fallback with the
  same command abstraction (↑ forward toggle, ↓ reverse, Space/T stop, ←→ steer, S center, A/Z speed, R
  record, M mark, Q/ESC quit). **The L1 dead-man was dropped on 2026-11-09** (user decision after the first real drive):
  `input.ds4.deadman` defaults to `false`; setting it to `true` restores "drive only while L1 is held". The no-input
  timer (`safety.deadman_sec`) applies to keyboard toggle driving only, never while a trigger is held.
- Safety: stop + center on SIGTERM/SIGINT and on evdev device loss; `deadman_sec` auto-stop; startup
  checks. **Three guards added 2026-11-10 after the car twice kept driving straight with a dead joystick:**
  (1) `_safe_write` polls the terminal for writability and drops the status line instead of blocking — a hung
  (not closed) SSH session fills the pty buffer in seconds and a blocking write would freeze the main loop while the
  PCA9685 holds the last command; (2) a watchdog thread (`WATCHDOG_SEC` = 1 s) stops the motor and quits if the main
  loop stops ticking for any reason; (3) `DS4Input` also reads the pad's hidraw node (`HidrawLink`): the DS4 streams
  ~450 HID reports/s even when idle, so `input.ds4.link_timeout_sec` (0.5 s) of silence means the Bluetooth link is dead
  → `DEVICE_LOST` → stop, instead of waiting ~20 s for the kernel to remove the device (needs udev rule
  `91-ds4-hidraw.rules`, installed by `jetson_setup.sh`). **SIGHUP is ignored when driving with the pad** (user decision 2026-11-10: SSH closing must not stop the car;
  the pad is the controller and the dot matrix shows status; the keyboard mode still stops on SIGHUP). A flock in
  `dataset/collector.lock` (`collector/lock.py`) refuses a second collector and prints the `kill <PID>` command. Startup
  checks for the I2C bus and both PCA9685 boards, camera, disk space.

## Curator behaviour (B2, confirmed 2026-11-04)

PySide6 desktop app. Open one or more session folders or zips; show one frame at a time with angle/speed
parsed from the filename and the steering angle drawn as an arrow (direction/scale from config, with an
invert flag); thumbnail filmstrip; ←/→ step, PageUp/PageDown ×10, Home/End, auto-play; `[` `]` mark a range
then Delete excludes it; Ctrl+Z undo; frames inside `marks.csv` intervals are pre-flagged; live per-class
counts and ratios; "데이터셋 만들기" writes the zip described above.

## Legacy notes (`legacy/`, untracked local reference only — also in git history before `d2e0de0`)

Old pipeline: `legacy/datacollector/img-collector.py` → `img-cleaner.py` → `legacy/training/train_pilotnet.py`
(PilotNet, 5-class classification, exports ONNX) → trtexec → `legacy/inference/run_inference.py`. It was
run from the old repo root as modules and will not run from `legacy/` without path changes; it is not meant
to be run any more. `legacy/preprocessor/RCPreprocessor.py` (crop top 40 %, resize 200×66, BGR→RGB, 0–1,
CHW) and the PilotNet definition in `legacy/training/model.py` are the pieces worth carrying over into
`common/` and `trainer/`. `legacy/requirements.txt` is UTF-16 encoded. `legacy/training/export_onnx.py`
imports a module that does not exist.
