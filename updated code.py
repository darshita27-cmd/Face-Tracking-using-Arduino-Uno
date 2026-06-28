import cv2 # read camera image, draw boxes, circles, detevt movements
import mediapipe as mp # find eyes and detect blinking
import numpy as np #go through tabular images
import pyfirmata2 as pyfirmata # to cotrol arduino
import time 
import os # create captures folder
import threading # threading helps with multiple tasks like: watching cam, moving servo, sending telegram messages
import asyncio # asyncio is another way of sending telegram while it keeps working insead of wait 
from datetime import datetime # current date and time
from scipy.spatial import distance as dist # need to measure distance between points eg:distance between eye one and two. need to detect blinking

# python-telegram-bot v20+ (async)
from telegram import Bot # creates a bot instance to send messages and photos
from telegram.error import TelegramError # handles errors when sending messages or photos

# MediaPipe 0.10.x task-based API
from mediapipe.tasks import python as mp_tasks # loads the mediapipe tasks module
from mediapipe.tasks.python import vision as mp_vision # loads vision tasks including the face landmarker
from mediapipe.tasks.python.vision import (
    FaceLandmarker, # find face points like eyes, nose, mouth etc
    FaceLandmarkerOptions, # settings for facelandmarker like confidence, no. of faces, tracking settings
)
# FaceLandmarkerResult lives in components, not vision — wrong submodule causes ImportError
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark  # represents a single landmark point with x,y,z coordinates normalized to [0,1]
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult 

# CONFIGURATION SETTINGS--------------------------------


WS, HS      = 1280, 720       # Camera frame 
CAM_INDEX   = 0               # 0-> laptop cam, 1-> USB webcam, 2-> another cam

ARDUINO_PORT  = "COM11"
SERVO_PIN_X   = 9             # Horizontal servo
SERVO_PIN_Y   = 10            # Vertical servo

SERVO_X_MIN, SERVO_X_MAX = 0, 180  # servo orizonatl limits
SERVO_Y_MIN, SERVO_Y_MAX = 0, 90  # servo y limits

# Liveness / EAR (report §3.3: threshold ~0.20, window 7 s, ≥1 blink)
EAR_THRESHOLD       = 0.20  # EAR=Ear Aspect Ratio. no. describes how open the eye is. Blinks cause it to drop sharply. Threshold of 0.20 is commonly used to detect blinks.
EAR_CONSEC_FRAMES   = 3  # frame 1 closed frame 2 closed frame 3 closed = blink. This helps filter out noise. eyes should be closed for 3 consecutive frames
LIVENESS_WINDOW_SEC = 7 # systme watches for 7 sec if in 7 secs if atleast 1 blink then its a real person if nnot then not than possibly a photo

# Motion detection
MIN_CONTOUR_AREA = 3000 # Minimum area of detected contour to be considered a person. Helps filter out small movements like curtains or pets.
BLUR_SIZE        = (21, 21) # Size of the Gaussian blur kernel applied to frames for motion detection. Larger values smooth more but may reduce sensitivity to small movements.
DILATE_ITER      = 2 # helps in filling small haoles. eg: brfore :  ####     ##### after: ###############  make moving objects easier to detect

# Background calibration / refresh settings
CALIBRATION_FRAMES      = 30    # Frames averaged for the initial background model
BACKGROUND_REFRESH_SEC  = 10    # Seconds before refreshing background when no person present

# Telegram configuration -----------------------------------

TELEGRAM_BOT_TOKEN  = "8908584956:AAG3acecp8Q99miC5DiZz2AJ96f2qAArSlE"    # bot's password
TELEGRAM_CHAT_ID    = "1265703075"      # chat id 
CAPTURE_DIR         = "captures"     
ALERT_COOLDOWN_SEC  = 60          # Minimum seconds between Telegram alerts 


# Servo thread configuration ---------------------------------------
SERVO_THREAD_INTERVAL  = 0.020   # 20 ms between servo updates — 50 Hz

# FIX 1: raised from 0.25 → 0.45 so servo closes 45% of the error per tick
# instead of 25%, reaching the target roughly 2× faster without overshooting
SMOOTH_ALPHA           = 0.45

# FIX 2: raised from 3.5 → 6.0 deg/tick — the old cap was the main bottleneck
# for large target jumps; 6.0 deg × 50 Hz = 300 deg/s which most hobby servos
# can sustain, so this is now realistic rather than artificially restrictive
MAX_SPEED_DEG_PER_TICK = 6.0


# the higher SMOOTH_ALPHA converging to values that toggled around the old threshold
SERVO_DEADBAND         = 0.5



# servo target; CENTROID_ALPHA=0.60 keeps ~40% of the previous position each frame.
# Raised from 0.35 → 0.60: the old value was the primary cause of tracking lag —
# it discarded 65% of the new position every frame, making the servo feel "sticky".
CENTROID_ALPHA = 0.60 # 0.35=smooth/slow, 0.60=responsive, 0.80=fast/jittery


# MediaPipe model
MODEL_PATH = "face_landmarker.task"  # trained model for face detection and blinking

# Eye landmark indices (MediaPipe Face Mesh 468-point topology)
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380] # to calculate EAR we need 6 points around the eye. these are the indices of those points in the 468-point face mesh. These specific points are chosen because they form a good representation of the eye's shape and can be used to calculate the EAR effectively.
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144] 


# EAR CALCULATION or say blinking math --------------------------------------

def eye_aspect_ratio(eye_landmarks):
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Drops sharply on blink; threshold ~0.20 (report §3.3).
    """
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5]) # one eye height (point 2 nd 6)
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4]) # other eye height (point 3 and 5)
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3]) # mearsure eye width (point 1 and 4)
    return (A + B) / (2.0 * C) # average height/width



# SHARED RESULT HOLDER  (LIVE_STREAM callback) the AI model used is asynchronous and the main program uses synchronous loop( grab frame, process fram, process motion, update servo etc). 
# When you call face_landmarker.detect_async(), your main loop doesn't wait around for the AI to finish calculating. Instead, your main loop immediately moves on to the next line of code so the video feed doesn't freeze.
# When the AI does finish calculating a few milliseconds later, it triggers a separate function called a callback function (result_callback) on a completely different background thread. Because result_callback runs on that background thread, it cannot directly "return" a value to your main loop. It needs a shared bucket where it can drop off the results. That bucket is latest_result.
#initialy no face found therefore None

latest_result = {"landmarks": None}

# after mediapie is done processing the frame, this function receives that message
def result_callback(
    result: FaceLandmarkerResult, # contains everything mediapie found (nose, eyes mouth, face landmarks)
    output_image, # the processed image sent by mediapie. code dosen't use it but mediapie sends it by default
    timestamp_ms: int, # timestamp of frame. Code dosen't use it either
):
    """Async callback: stores the most recent face landmark set."""
    latest_result["landmarks"] = result.face_landmarks[0] if result.face_landmarks else None #result.face_landmarks could contain [Face 1# landmarks] or [] if no face was found. result.face_landmarks[0] means 'give me first face' later in the code we used num_faces=1 so yup. so now...if result.face_landmarks means did the mediapipe find a face? if yes then give the first face if no then none. suppose if mediapipe found a face then 468 face points are stored


# Telegram alert helper-----------------


async def _send_telegram_photo_async(img_path: str, caption: str) -> None: # this function can do do waiting without blocking everything. eg:sends telegram while waiting: camera, servo continues. the image file and the caption or text sent with it
    
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    try:
        with open(img_path, "rb") as photo_file:
            await bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=photo_file,
                caption=caption,
            )
        print("[TG]  Image sent successfully via Telegram.")
    except TelegramError as e:
        print(f"[TG WARN] Telegram send failed: {e}")
    except FileNotFoundError:
        print(f"[TG WARN] Image file not found: {img_path}")


def send_telegram_alert(img_path: str, caption: str) -> None: # async can't work directly in the main loop because main loop is synchronous. so we create a new event loop just for this async function and run it until it completes. this way we can send telegram alerts without freezing the main loop.
    
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_send_telegram_photo_async(img_path, caption))
    finally:
        loop.close()


_alert_lock          = threading.Lock() # thread 1 tries tos end alerts. thread 2 tries to send alrert. can't send two at same time therefore lock is used such that only on can occupy
alert_in_flight      = False     # Protected by _alert_lock
last_alert_time      = 0.0       # Protected by _alert_lock
prev_liveness        = False     # Tracks previous frame's person_is_live value
person_was_detected  = False     # True if a person was present in the previous frame.
                                 # Used to detect genuine re-entry so the cooldown
                                 # is bypassed — a new arrival should always alert.


def _alert_worker(img_path: str, caption: str) -> None: # this works inside a seprate thread (try_send_alert) because sending telegram messages may take time. camera should not stop working while waiting for it
    
    global alert_in_flight # this means that real variable will be modified, not creating a new local variable
    try:
        send_telegram_alert(img_path, caption)
    finally:
        with _alert_lock:
            alert_in_flight = False



# this is the brain. every frame eventually reaches this function. as for parameters: frame (used if we decide to save a photo), person_is_live ( if true then blink detection, real person), new_entry (noboady present-> persons enters -> new entry true)
def try_send_alert(frame: np.ndarray, person_is_live: bool, new_entry: bool = False) -> None:
    
    global prev_liveness, last_alert_time, alert_in_flight # modifying the real variables

    # Rising-edge detection: only proceed if liveness just became True
    rising_edge = person_is_live and not prev_liveness # rising edge means the moment when person_is_live changes from false to true we want to send alert only at that movement not continously
    prev_liveness = person_is_live # storing the current state for next frame

    if not rising_edge:
        return

    now = time.time() # used for cooldown calculation

    with _alert_lock:  # _with alret_lock only one thread is allowed here
        
        cooldown_ok   = new_entry or (now - last_alert_time) >= ALERT_COOLDOWN_SEC # new_entry means if a person just left and a new person enters u probably need an alert therefore if that happens then don't wait for cooldown
        not_in_flight = not alert_in_flight # don't start another alret if one is is still sending

    if not (cooldown_ok and not_in_flight):
        reason = "cooldown active" if not cooldown_ok else "previous send still in flight"
        print(f"[ALERT] Rising edge detected but alert suppressed ({reason}).")
        return

    # Save image in the main thread (fast, no network I/O)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_filename  = os.path.join(CAPTURE_DIR, f"detected_{timestamp_str}.jpg")
    cv2.imwrite(img_filename, frame)
    print(f"[ALERT] Live person detected. Image saved: {img_filename}")

    caption = (
        f"[TURRET ALERT] Live person detected at "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
    )

    # Mark alert in-flight and update cooldown timer BEFORE starting the thread
    # so that even if the thread is slow, no second alert can slip through.
    with _alert_lock:
        alert_in_flight = True
        last_alert_time = now

    t = threading.Thread(   # creating a new thread
        target=_alert_worker,
        args=(img_filename, caption),
        daemon=True, # if the program exits, kill this thread immediately
        name="TelegramAlertThread",  # just a name for debugging purpose
    )
    t.start()
    print("[ALERT] Telegram alert thread started.")

_servo_lock         = threading.Lock()
_servo_target       = [90.0, 45.0] # like where want to be
_servo_current      = [90.0, 45.0] # like where i am. so that the servos moves gradually
_servo_last_written = [None, None]
_servo_thread_run   = True # should servo keep running ? true ? forever. false? stop servo thread


def _servo_thread_fn(pin_x, pin_y):
    
    global _servo_current, _servo_last_written

    while _servo_thread_run:
        t0 = time.time() # will be used later to maintain fixed update rate

        with _servo_lock:
            target = list(_servo_target) # copying cuz another thread might change. copying will avoid weird bugs

        for i, (pin, s_min, s_max) in enumerate(  # this one runs twic. once for i=0 means for x and i=1 for y
            [(pin_x, SERVO_X_MIN, SERVO_X_MAX),
             (pin_y, SERVO_Y_MIN, SERVO_Y_MAX)]
        ):
            error = target[i] - _servo_current[i]
            step  = SMOOTH_ALPHA * error # FIX 1 in effect: 45% of error closed per tick
            step  = float(np.clip(step, -MAX_SPEED_DEG_PER_TICK, MAX_SPEED_DEG_PER_TICK)) # FIX 2 in effect: cap is now 6.0 deg/tick
            new_pos = float(np.clip(_servo_current[i] + step, s_min, s_max))
            _servo_current[i] = new_pos

            if (_servo_last_written[i] is None or
                    abs(new_pos - _servo_last_written[i]) >= SERVO_DEADBAND):  # FIX 3 in effect: deadband is now 0.5 deg
                try:
                    pin.write(new_pos)
                    _servo_last_written[i] = new_pos
                except Exception as e:
                    print(f"[SERVO WARN] Write failed on axis {i}: {e}")

        elapsed   = time.time() - t0 # to calculate how much time the loop took
        remaining = SERVO_THREAD_INTERVAL - elapsed 
        if remaining > 0:
            time.sleep(remaining)


print("[INFO] Opening camera …")
cap = None  # cap=none means no camera connected yet
for _attempt in range(3):
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WS)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HS)
    time.sleep(2)
    if cap.isOpened():
        break
    print(f"[WARN] Camera open attempt {_attempt + 1} failed, retrying …")
    cap.release()

if not cap or not cap.isOpened():
    print("[ERROR] Camera could not be accessed after 3 attempts. Exiting.")
    exit(1)
print("[OK] Camera initialised.")


# ARDUINO / SERVO INITIALISATION-------------------------------------


try:
    board = pyfirmata.Arduino(ARDUINO_PORT)
    it    = pyfirmata.util.Iterator(board)
    it.start()
    print(f"[OK] Connected to Arduino on {ARDUINO_PORT}.")
except Exception as e:
    print(f"[ERROR] Failed to connect to Arduino on {ARDUINO_PORT}: {e}")
    cap.release()
    exit(1)

servo_pin_x = board.get_pin(f'd:{SERVO_PIN_X}:s')
servo_pin_y = board.get_pin(f'd:{SERVO_PIN_Y}:s')

servo_pin_x.write(_servo_current[0])
servo_pin_y.write(_servo_current[1])
_servo_last_written = [_servo_current[0], _servo_current[1]]

servo_thread = threading.Thread( # creating a thread
    target=_servo_thread_fn, # the work for this thread
    args=(servo_pin_x, servo_pin_y), # expected input
    daemon=True, # servo dies automatically no clean up needed after exiting the program
    name="ServoControlThread",
)
servo_thread.start()
print("[OK] Servo control thread started (50 Hz, velocity-limited).")


# ─────────────────────────────────────────────
# MEDIAPIPE FACE LANDMARKER  (v0.10.x API)
# ─────────────────────────────────────────────

if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model file '{MODEL_PATH}' not found.")
    print("  Download with:")
    print("  python -c \"import requests; open('face_landmarker.task','wb').write("
          "requests.get('https://storage.googleapis.com/mediapipe-models/"
          "face_landmarker/face_landmarker/float16/1/face_landmarker.task').content)\"")
    cap.release()
    exit(1)

base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
options = FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.LIVE_STREAM, # Instead of processing static images or video files frame-by-frame, LIVE_STREAM tells the model to run asynchronously. If a frame takes too long to process, it drops it and instantly hops to the newest frame so your physical servos never experience delay.
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=result_callback,
)
face_landmarker = FaceLandmarker.create_from_options(options)
print("[OK] MediaPipe FaceLandmarker initialised.")


# ─────────────────────────────────────────────
# CAPTURE DIRECTORY
# ─────────────────────────────────────────────

os.makedirs(CAPTURE_DIR, exist_ok=True) # capture directory. if dosen't exsist then creats it 


# ─────────────────────────────────────────────
# STATE VARIABLES
# ─────────────────────────────────────────────

# Background calibration state [BUG-A FIX retained]
background_frame       = None
calibration_count      = 0 # target is of 30 frames. in this +1 will happen everytime vide frame is processed
calibration_accum      = None  #A floating-point grid math bucket. Because adding 30 images together results in very large numbers, standard image formats would overflow and corrupt. This variable safely accumulates the raw pixel math before the script calculates the average.
background_calibrated  = False
last_bg_refresh_time   = time.time() # used to refress after 10 secdonds such that natural light or such changes don't confuse motion sensor

blink_counter         = 0 # to calculate the 3 consecutive frames for eye shut
blink_total           = 0 
liveness_window_start = time.time()
person_is_live        = False

# FIX 4: smoothed centroid state — starts at frame centre so the servo doesn't
# lurch to (0,0) on the very first detection
_cx_smooth = float(WS // 2)
_cy_smooth = float(HS // 2)


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

print("[INFO] System running. Press 'q' to quit.")
print(f"[INFO] Calibrating background … ({CALIBRATION_FRAMES} frames needed)")

while True:
    success, frame = cap.read() # sucess id true or false to see if the camera was successfully open. frame is the actual matrix in BGR format
    if not success or frame is None:
        print("[WARN] Skipping unreadable frame.")
        continue

    display = frame.copy()

    # [FIX-1] Real wall-clock timestamp for MediaPipe LIVE_STREAM mode.
    frame_ts_ms = int(time.time() * 1000)

    # ── PERCEPTION LAYER: Contour-based motion detection ──────────────────

    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, BLUR_SIZE, 0)

    # [BUG-A FIX] Background calibration and frozen-background logic
    if not background_calibrated:
        if calibration_accum is None:
            calibration_accum = np.float32(blurred)
        else:
            cv2.accumulate(blurred, calibration_accum)
        calibration_count += 1

        cv2.putText(display,
                    f"CALIBRATING … {calibration_count}/{CALIBRATION_FRAMES}",
                    (30, 50), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 200, 255), 3)
        cv2.imshow("Turret System – Darshita Singh 229310009", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if calibration_count >= CALIBRATION_FRAMES:
            background_frame      = (calibration_accum / CALIBRATION_FRAMES).astype(np.uint8)
            background_calibrated = True
            print("[OK] Background calibration complete.")
        continue

    frame_delta = cv2.absdiff(background_frame, blurred)
    thresh      = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh      = cv2.dilate(thresh, None, iterations=DILATE_ITER)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    person_detected = False
    person_cx, person_cy = WS // 2, HS // 2

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)
        if area >= MIN_CONTOUR_AREA:
            person_detected = True
            bx, by, bw, bh = cv2.boundingRect(largest)
            person_cx = bx + bw // 2
            person_cy = by + bh // 2
            # Green 16x16 rectangle and dot are drawn later, anchored to the face
        else:
            print(f"[DEBUG] Contour area {int(area)} below MIN_CONTOUR_AREA "
                  f"{MIN_CONTOUR_AREA} — ignored. [FIX-10]")

    # Refresh background only when nobody present [BUG-A FIX]
    if not person_detected:
        now = time.time()
        if (now - last_bg_refresh_time) >= BACKGROUND_REFRESH_SEC:
            background_frame  = cv2.addWeighted(background_frame, 0.9, blurred, 0.1, 0)
            last_bg_refresh_time = now

        person_is_live             = False
        blink_counter              = 0
        blink_total                = 0
        liveness_window_start      = time.time()
        latest_result["landmarks"] = None

        # FIX 4: reset smoothed centroid to frame centre when target is lost
        # so the servo returns to neutral and doesn't drift on re-acquisition
        _cx_smooth = float(WS // 2)
        _cy_smooth = float(HS // 2)

    # ── INTELLIGENCE LAYER: MediaPipe FaceLandmarker EAR blink detection ──

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
    )
    face_landmarker.detect_async(mp_image, frame_ts_ms)

    face_lms = latest_result["landmarks"]

    if face_lms is not None:
        h, w = frame.shape[:2]

        def get_eye_coords(indices):
            return [(int(face_lms[i].x * w), int(face_lms[i].y * h))
                    for i in indices]

        left_eye  = get_eye_coords(LEFT_EYE_IDX)
        right_eye = get_eye_coords(RIGHT_EYE_IDX)

        left_ear  = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear   = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= EAR_CONSEC_FRAMES:
                blink_total += 1
                # Immediately mark as live on first confirmed blink so the alert
                # fires right away instead of waiting for the 7-second window end.
                person_is_live = True
            blink_counter = 0

        elapsed = time.time() - liveness_window_start
        if elapsed >= LIVENESS_WINDOW_SEC:
            person_is_live        = (blink_total >= 1)   # [FIX-3]
            blink_total           = 0                     # [FIX-5]
            blink_counter         = 0
            liveness_window_start = time.time()

        for pt in left_eye + right_eye:
            cv2.circle(display, pt, 2, (255, 0, 255), -1)

        cv2.putText(display, f"EAR: {avg_ear:.2f}",    (WS - 220, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 255, 0), 2)
        cv2.putText(display, f"Blinks: {blink_total}", (WS - 220, 80),
                    cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 255, 0), 2)

    # ── ACTION LAYER: Servo target update ─────────────────────────────────

    # FIX 5: when a face is detected by MediaPipe, use the NOSE TIP position as
    # the servo target instead of the body-contour centroid.
    # WHY: the contour bounding box covers the whole body (often 400x600 px), so
    # the centroid barely shifts when the person moves — the servo "sleeps".
    # The nose tip is a single precise pixel-level point: any head movement causes
    # a much larger angular change, making the servo snap to position faster.
    # Fall back to the body contour centroid only when no face is in frame.
    if person_detected:
        if face_lms is not None:
            h_f, w_f = frame.shape[:2]
            face_cx = int(face_lms[1].x * w_f)   # landmark 1 = nose tip
            face_cy = int(face_lms[1].y * h_f)
            _cx_smooth = CENTROID_ALPHA * face_cx + (1 - CENTROID_ALPHA) * _cx_smooth
            _cy_smooth = CENTROID_ALPHA * face_cy + (1 - CENTROID_ALPHA) * _cy_smooth
        else:
            # No face found yet — fall back to body contour centroid
            _cx_smooth = CENTROID_ALPHA * person_cx + (1 - CENTROID_ALPHA) * _cx_smooth
            _cy_smooth = CENTROID_ALPHA * person_cy + (1 - CENTROID_ALPHA) * _cy_smooth
    # (no else needed — reset already handled in the no-person block above)

    target_x = float(np.interp(_cx_smooth, [0, WS], [SERVO_X_MAX, SERVO_X_MIN]))
    target_y = float(np.interp(_cy_smooth, [0, HS], [SERVO_Y_MAX, SERVO_Y_MIN]))

    with _servo_lock:
        _servo_target[0] = target_x
        _servo_target[1] = target_y

    
    new_entry = person_detected and not person_was_detected
    person_was_detected = person_detected

    if person_detected:
        try_send_alert(frame, person_is_live, new_entry=new_entry)
    else:
        # Person left the frame — keep prev_liveness in sync with liveness state
        # so the next re-entry generates a proper rising edge (False→True).
        if not person_is_live:
            prev_liveness = False

    # ── DISPLAY OVERLAYS ──────────────────────────────────────────────────

    cv2.line(display, (0, HS // 2), (WS, HS // 2), (200, 200, 200), 1)
    cv2.line(display, (WS // 2, 0), (WS // 2, HS), (200, 200, 200), 1)

    # 256×256 green rectangle — follows face centre only when face is detected
    if face_lms is not None:
        h_d, w_d = frame.shape[:2]
        rect_cx = int(face_lms[1].x * w_d)   # landmark 1 = nose tip (face centre X)
        rect_cy = int(face_lms[1].y * h_d)
        half = 128  # half of 256 px
        cv2.rectangle(
            display,
            (rect_cx - half, rect_cy - half),
            (rect_cx + half, rect_cy + half),
            (0, 255, 0), 2,
        )

    if person_detected:
        # Use face centre for the red tracking circle when a face is available;
        # face centre = midpoint of forehead (lm 10) and chin (lm 152) vertically,
        # nose tip (lm 1) horizontally — much more accurate than the body centroid.
        if face_lms is not None:
            h_d, w_d = frame.shape[:2]
            track_cx = int(face_lms[1].x * w_d)                                      # nose tip X
            track_cy = int((face_lms[10].y + face_lms[152].y) / 2.0 * h_d)          # forehead–chin midpoint Y
        else:
            track_cx, track_cy = person_cx, person_cy  # fallback: body contour centroid
        cv2.line(display,   (0, track_cy),        (WS, track_cy),       (0, 0, 0), 2)
        cv2.line(display,   (track_cx, 0),         (track_cx, HS),       (0, 0, 0), 2)
        cv2.circle(display, (track_cx, track_cy), 15, (0, 0, 255), cv2.FILLED)
        cv2.circle(display, (track_cx, track_cy), 80, (0, 0, 255), 2)
        status_text  = "LIVE - TARGET LOCKED" if person_is_live else "TARGET LOCKED"
        status_color = (0, 255, 0)             if person_is_live else (0, 140, 255)
        cv2.putText(display, status_text, (30, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2.5, status_color, 3)
    else:
        cv2.circle(display, (WS // 2, HS // 2), 80, (0, 0, 255), 2)
        cv2.circle(display, (WS // 2, HS // 2), 15, (0, 0, 255), cv2.FILLED)
        cv2.line(display,   (0, HS // 2),    (WS, HS // 2),    (0, 0, 0), 2)
        cv2.line(display,   (WS // 2, 0),    (WS // 2, HS),    (0, 0, 0), 2)
        cv2.putText(display, "NO TARGET", (30, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 255), 3)

    with _servo_lock:
        disp_x = int(_servo_current[0])
        disp_y = int(_servo_current[1])

    cv2.putText(display, f"Servo X: {disp_x} deg", (30, HS - 60),
                cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 0, 0), 2)
    cv2.putText(display, f"Servo Y: {disp_y} deg", (30, HS - 30),
                cv2.FONT_HERSHEY_PLAIN, 1.8, (255, 0, 0), 2)

    
    if person_detected:
        live_label = "LIVE"      if person_is_live else "VERIFYING..."
        live_color = (0, 255, 0) if person_is_live else (0, 165, 255)
        cv2.putText(display, f"Liveness: {live_label}", (30, 90),
                    cv2.FONT_HERSHEY_PLAIN, 1.8, live_color, 2)

    cv2.imshow("Turret System – Darshita Singh 229310009", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting on user command.")
        break


# ─────────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────────

_servo_thread_run = False
servo_thread.join(timeout=1.0)
cap.release()
cv2.destroyAllWindows()
face_landmarker.close()
print("[INFO] Resources released. System stopped.")