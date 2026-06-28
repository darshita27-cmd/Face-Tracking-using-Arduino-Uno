# Face Tracking Turret System — Arduino Uno

Real-time face tracking and liveness detection system using **MediaPipe**, **OpenCV**, and **Arduino** with servo motors. Detects motion, verifies live presence via blink detection (EAR), controls a pan-tilt servo rig, and sends Telegram photo alerts when a real person is confirmed.

---

## Features

- **Contour-based motion detection** with adaptive background calibration (30-frame average, auto-refresh every 10 s when idle)
- **MediaPipe FaceLandmarker** (v0.10.x task API, 468-point mesh) for precise face localisation — replaces the older Haar Cascade / CVZone approach
- **Liveness detection via Eye Aspect Ratio (EAR)** — confirms a real person by detecting ≥ 1 blink within a 7-second rolling window; rejects printed photos and static video feeds
- **Servo pan-tilt tracking** driven by the face's nose-tip landmark (landmark 1) for pixel-precise targeting; falls back to body-contour centroid when no face is detected
- **Velocity-limited, low-latency servo control** running on a dedicated 50 Hz background thread with:
  - Exponential smoothing (`SMOOTH_ALPHA = 0.45`)
  - Speed cap (`MAX_SPEED_DEG_PER_TICK = 6.0 deg/tick`)
  - Deadband filtering (`SERVO_DEADBAND = 0.5 deg`) to prevent jitter
  - Responsive centroid tracking (`CENTROID_ALPHA = 0.60`)
- **Telegram photo alerts** — saves a timestamped JPEG and sends it via bot when a live person is first confirmed; cooldown of 60 s between alerts; bypassed on genuine new entry
- **Threaded architecture** — main loop, servo thread, and Telegram alert thread run concurrently; MediaPipe inference runs asynchronously (LIVE_STREAM mode)
- **On-screen HUD** — crosshairs, EAR value, blink count, servo angles, liveness status (LIVE / VERIFYING… / NO TARGET), and 256 × 256 face bounding box

---

## Repository Structure

```
Face-Tracking-using-Arduino-Uno/
├── updated code.py                    # Main application (run this)
├── face_tracking.ipynb                # Original Jupyter notebook (reference)
├── track.ipynb                        # Experimental tracking notebook
├── haarcascade_frontalface_default.xml# Legacy Haar cascade (no longer used by main script)
├── requirements.txt                   # Python dependencies
├── captures/                          # Auto-created; stores alert snapshots
└── face_landmarker.task               # MediaPipe model — download before running (see below)
```

---

## Hardware Requirements

- Webcam (built-in or USB)
- Arduino Uno (or Mega)
- 2 × servo motors
- Jumper wires
- USB cable for Arduino

---

## Software Requirements

### Python

- Python 3.9 – 3.11 (MediaPipe wheels are not yet available for 3.12+)

### Libraries

Install all dependencies:

```bash
pip install opencv-python mediapipe scipy numpy pyfirmata2 python-telegram-bot
```

| Package | Purpose |
|---|---|
| `opencv-python` | Frame capture, image processing, display |
| `mediapipe` | Face landmark detection (468-point mesh), EAR blink detection |
| `scipy` | Euclidean distance for EAR calculation |
| `numpy` | Frame arithmetic, servo interpolation |
| `pyfirmata2` | Arduino serial communication / servo control |
| `python-telegram-bot` | Async Telegram bot alerts (v20+) |

### MediaPipe Model File

Download `face_landmarker.task` into the project root before running:

```bash
python -c "
import requests
open('face_landmarker.task', 'wb').write(
    requests.get(
        'https://storage.googleapis.com/mediapipe-models/'
        'face_landmarker/face_landmarker/float16/1/face_landmarker.task'
    ).content
)
"
```

### Arduino Firmware

Upload **StandardFirmata** to your Arduino:

1. Open Arduino IDE
2. Go to **File → Examples → Firmata → StandardFirmata**
3. Upload to your board

---

## Wiring

| Servo | Arduino Pin |
|---|---|
| Horizontal (X-axis) | Digital 9 |
| Vertical (Y-axis) | Digital 10 |

Connect servo power and ground as appropriate for your servo's voltage rating.

---

## Configuration

All tuneable parameters are at the top of `updated code.py`:

| Parameter | Default | Description |
|---|---|---|
| `WS, HS` | `1280, 720` | Camera resolution |
| `CAM_INDEX` | `0` | Camera index (`0` = built-in, `1` = USB, etc.) |
| `ARDUINO_PORT` | `"COM11"` | Serial port of your Arduino |
| `SERVO_PIN_X` | `9` | Horizontal servo pin |
| `SERVO_PIN_Y` | `10` | Vertical servo pin |
| `SERVO_X_MIN/MAX` | `0, 180` | Horizontal servo range (degrees) |
| `SERVO_Y_MIN/MAX` | `0, 90` | Vertical servo range (degrees) |
| `EAR_THRESHOLD` | `0.20` | Eye Aspect Ratio threshold for blink detection |
| `EAR_CONSEC_FRAMES` | `3` | Consecutive sub-threshold frames to register one blink |
| `LIVENESS_WINDOW_SEC` | `7` | Rolling window (seconds) for blink counting |
| `MIN_CONTOUR_AREA` | `3000` | Minimum pixel area to count as a person |
| `BLUR_SIZE` | `(21, 21)` | Gaussian blur kernel for motion detection |
| `CALIBRATION_FRAMES` | `30` | Frames averaged to build the background model |
| `BACKGROUND_REFRESH_SEC` | `10` | Seconds before refreshing background when idle |
| `SMOOTH_ALPHA` | `0.45` | Servo exponential smoothing factor |
| `MAX_SPEED_DEG_PER_TICK` | `6.0` | Max servo movement per tick (degrees) |
| `SERVO_DEADBAND` | `0.5` | Minimum angle change to trigger a servo write |
| `CENTROID_ALPHA` | `0.60` | Tracking centroid smoothing (higher = more responsive) |
| `ALERT_COOLDOWN_SEC` | `60` | Minimum seconds between Telegram alerts |
| `TELEGRAM_BOT_TOKEN` | — | Your bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | — | Your Telegram chat / user ID |

> **Before running:** replace `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` with your own credentials. Never commit real tokens to a public repo.

---

## Usage

1. Connect your Arduino and note its COM port (update `ARDUINO_PORT` in the script).
2. Connect both servo motors to pins 9 and 10.
3. Ensure `face_landmarker.task` is in the project root.
4. Run the script:

```bash
python "updated code.py"
```

5. The system will **calibrate the background** for 30 frames (a progress indicator is shown on screen).
6. Once calibrated, it enters the main tracking loop:
   - Motion triggers contour detection → servo follows.
   - MediaPipe confirms face → servo locks onto nose tip.
   - Blink is detected within 7 s → status changes to **LIVE — TARGET LOCKED** and a Telegram alert fires.

**Controls**

| Key | Action |
|---|---|
| `q` | Quit the application |

---

## How It Works

### Architecture Overview

```
Main Loop (synchronous, per-frame)
│
├── Motion Detection  ──  Background subtraction + contour finding
├── MediaPipe async   ──  LIVE_STREAM mode (non-blocking inference)
│        └── result_callback  ──  Updates latest_result["landmarks"]
│
├── EAR Blink Check   ──  Computes Eye Aspect Ratio on 6 landmarks per eye
│
├── Servo Target      ──  Nose tip (lm 1) → interp to servo degrees
│        └── _servo_lock  ──  Passes target to servo thread
│
└── Alert Decision    ──  Rising-edge liveness → spawns Telegram thread

ServoControlThread (50 Hz, daemon)
└── Reads target, applies smoothing + speed cap, writes to Arduino pins

TelegramAlertThread (on-demand, daemon)
└── Sends JPEG snapshot via python-telegram-bot async API
```

### Eye Aspect Ratio (EAR)

```
EAR = (||p2−p6|| + ||p3−p5||) / (2 × ||p1−p4||)
```

Six landmark points per eye define two vertical distances and one horizontal distance. When the eye closes, EAR drops sharply below `EAR_THRESHOLD` (0.20). Three consecutive sub-threshold frames register a confirmed blink.

### Servo Smoothing

On every 20 ms tick the servo thread computes:

```
step  = SMOOTH_ALPHA × (target − current)      # 45% of error per tick
step  = clip(step, −MAX_SPEED, +MAX_SPEED)     # cap at ±6 deg/tick
new   = clip(current + step, s_min, s_max)
```

A write is only sent to the Arduino when the position change exceeds `SERVO_DEADBAND` (0.5°), eliminating PWM noise when the target is nearly reached.

---

## Troubleshooting

**Camera not detected**
- Try changing `CAM_INDEX` to `1` or `2`.
- Ensure no other application is using the camera.
- On Windows, `cv2.CAP_DSHOW` is used automatically.

**Arduino connection failed**
- Verify the COM port (Windows Device Manager → Ports).
- Confirm StandardFirmata is uploaded to the board.
- Make sure no other process (e.g. Arduino IDE Serial Monitor) has the port open.

**`face_landmarker.task` not found**
- Run the download command in the [MediaPipe Model File](#mediapipe-model-file) section above.

**Servo not moving / moving erratically**
- Check servo power supply — servos should not be powered through the Arduino 5V pin under load.
- Reduce `SMOOTH_ALPHA` (e.g. `0.25`) or `MAX_SPEED_DEG_PER_TICK` (e.g. `3.0`) if the servo overshoots.

**No Telegram alerts received**
- Confirm `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are correct.
- Send `/start` to your bot in Telegram before running the script.
- Check console for `[TG WARN]` messages.

**Liveness always shows VERIFYING...**
- Improve lighting — EAR needs clear eye visibility.
- Blink naturally; fast or partial blinks may not meet `EAR_CONSEC_FRAMES = 3`.
- Lower `EAR_THRESHOLD` slightly (e.g. `0.18`) if your eyes are naturally narrow.

**Background calibration produces noisy motion**
- Increase `CALIBRATION_FRAMES` (e.g. `60`) or ensure the scene is static during startup.
- Increase `MIN_CONTOUR_AREA` to filter out small irrelevant movements.

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change, then submit a pull request.

