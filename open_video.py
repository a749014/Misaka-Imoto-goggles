import os
import glob
import cv2
import numpy as np
from datetime import datetime
from functools import partial
import time
from threading import Lock, Thread
from queue import Queue

from video_recorder import VideoRecorder
from gaussianeffect import GaussianEffect
from find_counters import MultipleFilters
from recognition import Model, format_model_menu_label, get_model_names

from menu_bar import GaussianblurMenu, SingleColorMenu, button, mark_button, render_buttons

try:
    import v4l2cam
except ImportError:
    print("[Warning] v4l2cam module not found. Falling back to OpenCV VideoCapture.")
    v4l2cam = None

#no fucking change , or this program will fuck you . fucking shit mountain 
BASE_WINDOW_WIDTH = 800
BASE_WINDOW_HEIGHT = 600
#change these when you want to change window size
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

SCALE_X = WINDOW_WIDTH / BASE_WINDOW_WIDTH
SCALE_Y = WINDOW_HEIGHT / BASE_WINDOW_HEIGHT
SCALE_MIN = min(SCALE_X, SCALE_Y)

threads = max(1, (os.cpu_count() or 4) - 1)
print(cv2.getNumThreads(), threads)
if cv2.getNumThreads() != threads:
    cv2.setNumThreads(threads)
print(cv2.ocl.haveOpenCL())      # 驱动是否存在
cv2.ocl.setUseOpenCL(True)
print(cv2.ocl.useOpenCL())       # 应为 True
print(cv2.ocl.Device.getDefault().name())  #
def scale_x(value: float) -> int:
    """Scale a horizontal value from the 800x600 baseline to the current window width."""
    return max(1, int(round(value * SCALE_X)))


def scale_y(value: float) -> int:
    """Scale a vertical value from the 800x600 baseline to the current window height."""
    return max(1, int(round(value * SCALE_Y)))


def scale_ui(value: float) -> int:
    """Scale a UI dimension using the smaller of the horizontal/vertical ratios."""
    return max(1, int(round(value * SCALE_MIN)))


def load_menu_settings(settings_path: str = "settings.conf") -> dict[str, float]:
    """Read menu-specific settings from the project config file.

    Args:
        settings_path: Path to the key=value config file.

    Returns:
        A dictionary with the menu hide interval in seconds.

    Example:
        menu_settings = load_menu_settings()
        hide_seconds = menu_settings["menu_hide_seconds"]
    """
    settings: dict[str, float] = {"menu_hide_seconds": 0.0}
    if not os.path.exists(settings_path):
        return settings

    with open(settings_path, "r", encoding="utf-8") as config_file:
        for line in config_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = (part.strip() for part in line.split("=", 1))
            if key != "menu_hide_seconds":
                continue
            try:
                settings[key] = float(value)
            except ValueError:
                settings[key] = 0.0
    return settings


currentIem = "Capture"
current_filter = "default"
menu_settings = load_menu_settings()
menu_hide_seconds = float(menu_settings.get("menu_hide_seconds", 0.0))
menu_visible = True
menu_last_action_time = time.time()
recording_active = False
recording_paused = False
recording_start_time = 0.0
recording_elapsed_accumulated = 0.0
video_recorder: VideoRecorder | None = None


def build_video_path() -> str:
    """Generate a timestamped video file path inside the gallery directory."""
    gallery_dir = os.path.join(os.getcwd(), "gallery")
    if not os.path.exists(gallery_dir):
        os.makedirs(gallery_dir, exist_ok=True)
    extension = str(menu_settings.get("video_format", "mp4")).strip().lstrip(".") or "mp4"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(gallery_dir, f"video_{timestamp}.{extension}")


def start_recording(frame: np.ndarray) -> None:
    """Begin recording using the configured codec and camera size."""
    global recording_active, recording_paused, recording_start_time, recording_elapsed_accumulated, video_recorder
    if recording_active and video_recorder is not None:
        return
    codec = str(menu_settings.get("video_codec", "mp4v")).strip() or "mp4v"
    video_recorder = VideoRecorder(
        build_video_path(),
        codec=codec,
        width=int(frame.shape[1]),
        height=int(frame.shape[0]),
        fps=max(float(fps), 1.0),
    )
    recording_active = True
    recording_paused = False
    recording_start_time = time.time()
    recording_elapsed_accumulated = 0.0
    print(f"[Video] recording started: {video_recorder.save_path}")


def pause_recording() -> None:
    """Pause or resume the active recording while keeping the timer state."""
    global recording_paused, recording_start_time, recording_elapsed_accumulated
    if not recording_active or video_recorder is None:
        return
    if recording_paused:
        recording_paused = False
        recording_start_time = time.time() - recording_elapsed_accumulated
        print("[Video] resumed")
    else:
        recording_paused = True
        recording_elapsed_accumulated = time.time() - recording_start_time
        print("[Video] paused")


def stop_recording() -> None:
    """Stop the current recording and save the file."""
    global recording_active, recording_paused, recording_start_time, recording_elapsed_accumulated, video_recorder
    recording_active = False
    recording_paused = False
    recording_start_time = 0.0
    recording_elapsed_accumulated = 0.0
    if video_recorder is None:
        return
    video_recorder.save()
    print(f"[Video] saved: {video_recorder.save_path}")
    video_recorder = None


def save_photo(frame: np.ndarray) -> str | None:
    """Save the current camera frame into the gallery directory.

    Args:
        frame: BGR camera frame to store as a JPEG file.

    Returns:
        The written image path, or None when the save operation fails.

    Example:
        photo_path = save_photo(frame)
    """
    gallery_dir = os.path.join(os.getcwd(), "gallery")
    os.makedirs(gallery_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_path = os.path.join(gallery_dir, f"photo_{timestamp}.jpg")
    if cv2.imwrite(save_path, frame):
        return save_path
    return None


def select_menu_item(item_name: str) -> None:
    """Store the name of the confirmed camera-page menu item.

    Parameters:
        item_name: Button label selected by the user.

    Returns:
        None. The global ``currentIem`` value is updated in place.

    Example:
        select_menu_item("Capture")
    """
    global currentIem
    currentIem = item_name


class CameraStream:
    """Read frames in a background thread and keep the newest frame available."""

    def __init__(self, src) -> None:
        self._uses_v4l2cam = False
        self._opened = False
        self.cap = None
        if isinstance(src, int) and v4l2cam is not None:
            print("finding v4l2 cameras")
            for device_path in sorted(glob.glob("/dev/video*")):
                camera = None
                try:
                    camera = v4l2cam.V4L2Camera(device_path, WINDOW_WIDTH, WINDOW_HEIGHT)
                    if camera.init() and camera.start():
                        self.cap = camera
                        self._uses_v4l2cam = True
                        self._opened = True
                        print(f"[CameraStream] Opened v4l2cam source {device_path}")
                        break
                except Exception as error:
                    print(f"[CameraStream] Could not open {device_path} with v4l2cam: {error}")
                finally:
                    if camera is not None and self.cap is None:
                        camera.stop()

        if self.cap is None:
            fallback_source = src
            if isinstance(src, int):
                video_devices = sorted(glob.glob("/dev/video*"))
                fallback_source = video_devices[0] if video_devices else src
            pipeline = (
                f"v4l2src device={fallback_source} ! "
                f"video/x-raw,width={WINDOW_WIDTH},height={WINDOW_HEIGHT} ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=true max-buffers=2"
            )
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER) if isinstance(src, int) else cv2.VideoCapture(src)
            self._opened = self.cap.isOpened()
            if not self._opened:
                print(f"[CameraStream] Failed to open source {fallback_source} with cap gstreamer, trying fallback to OpenCV VideoCapture.")
                self.cap = cv2.VideoCapture(fallback_source, cv2.CAP_V4L2) if isinstance(src, int) else cv2.VideoCapture(src)
            print(f"[CameraStream] Opened source {fallback_source}: {self._opened}")

        self.ret = False
        self._frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        self._lock = Lock()
        self._running = True
        self._thread = Thread(target=self._read_loop, daemon=True)
        self.fps = 0.0
        self._last_frame_time = None
        if isinstance(src, int) and not self._uses_v4l2cam:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            # self.cap.set(cv2.CAP_PROP_FPS, 30)
            print(f"[CameraStream] Camera properties: FPS={self.cap.get(cv2.CAP_PROP_FPS)}, Width={self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, Height={self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        self._thread.start()

    def _read_loop(self) -> None:
        while self._running:
            start_time = time.perf_counter()
            if self._uses_v4l2cam:
                try:
                    frame = self.cap.get_frame()
                    ret = frame is not None
                except Exception as error:
                    print(f"[CameraStream] v4l2cam read error: {error}")
                    ret, frame = False, None
            else:
                ret, frame = self.cap.read()
            # print(f"[CameraStream] Frame read: ret={ret}, shape={frame.shape if ret else 'N/A'}")
            with self._lock:
                self.ret = bool(ret)
                if ret:
                    self._frame = frame.get().copy() if isinstance(frame, cv2.UMat) else frame.copy()
                    if self._last_frame_time is not None:
                        frame_interval = start_time - self._last_frame_time
                        if frame_interval > 0:
                            self.fps = 1.0 / frame_interval
                    self._last_frame_time = start_time

    def read(self):
        with self._lock:
            return self.ret, self._frame.copy()

    def get(self, prop_id: int):
        if self._uses_v4l2cam:
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return WINDOW_WIDTH
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return WINDOW_HEIGHT
            return 0.0
        return self.cap.get(prop_id)

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=2)
        if self._uses_v4l2cam:
            self.cap.stop()
        else:
            self.cap.release()
        self.ret = False

    def isOpened(self) -> bool:
        return self._opened


def launch_gallery() -> None:
    """Close the camera, start the standalone gallery, and reopen the camera."""
    global stream, frame
    if stream is not None:
        stream.release()
    cv2.destroyAllWindows()
    from gallery import main as gallery_main
    gallery_main()
    cv2.destroyAllWindows()
    stream = CameraStream(0)
    if not stream.isOpened():
        raise RuntimeError("Failed to reopen camera after gallery exit")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)


def apply_filter_to_frame(frame: np.ndarray, filter_name: str) -> np.ndarray:
    """Apply the selected filter effect from MultipleFilters to the current camera frame."""
    filter_processor = MultipleFilters.__new__(MultipleFilters)
    filter_processor.image = frame.copy()

    if filter_name == "Sobel":
        return filter_processor.sobel_operator(frame.copy(), dx=1, dy=0, x_weight=1.0, y_weight=1.0)
    if filter_name == "Scharr":
        return filter_processor.scharr_operator(frame.copy(), dx=1, dy=0, x_weight=1.0, y_weight=1.0)
    if filter_name == "Sharpen":
        return filter_processor.sharpen_filter(frame.copy())
    if filter_name == "Relief":
        return filter_processor.relief_filter(frame.copy())
    if filter_name == "Old Photo":
        return filter_processor.old_photo_filter(frame.copy(), blur_ksize=(5, 5))
    if filter_name == "Sketch":
        return filter_processor.sketch_filter(frame.copy(), blur_ksize=(19, 19), scale=256)
    return frame

main_menu_base_size = {
    "width": 100,
    "position": (750, 330),
    "length": 600,
    "button_spacing": 4,
}
video_submenu_base_size = {
    "width": 90,
    "position": (main_menu_base_size["position"][0] - 120, main_menu_base_size["position"][1]),
    "length": 180,
    "corner_radius": 18,
}
filter_submenu_base_size = {
    "width": 90,
    "position": (650, 330),
    "length": 600,
    "corner_radius": 18,
}
recognition_submenu_base_size = {
    "width": 110,
    "position": (650, 330),
    "length": 600,
    "corner_radius": 18,
}

main_menu = SingleColorMenu(
    width=scale_x(main_menu_base_size["width"]),
    position=(scale_x(main_menu_base_size["position"][0]), scale_y(main_menu_base_size["position"][1])),
    length=scale_y(main_menu_base_size["length"]),
    rounded=False,
    button_spacing=scale_ui(main_menu_base_size["button_spacing"]),
    color=(127, 127, 127),
    orientation="vertical",
)
# print(f"Main menu position: {main_menu.position}, dimensions: width={main_menu.width}, length={main_menu.length}")
video_submenu = SingleColorMenu(
    width=scale_x(video_submenu_base_size["width"]),
    position=(scale_x(video_submenu_base_size["position"][0]), scale_y(video_submenu_base_size["position"][1])),
    length=scale_y(video_submenu_base_size["length"]),
    rounded=True,
    corner_radius=scale_ui(video_submenu_base_size["corner_radius"]),
    button_spacing=scale_ui(8),
    color=(110, 110, 110),
    orientation="vertical",
)
filter_submenu = SingleColorMenu(
    width=scale_x(filter_submenu_base_size["width"]),
    position=(scale_x(filter_submenu_base_size["position"][0]), scale_y(filter_submenu_base_size["position"][1])),
    length=scale_y(filter_submenu_base_size["length"]),
    rounded=True,
    corner_radius=scale_ui(filter_submenu_base_size["corner_radius"]),
    button_spacing=scale_ui(8),
    color=(90, 120, 160),
    orientation="vertical",
)
recognition_submenu = None
menu_items = ("Capture", "Record", "Recognize", "OCR", "amap api", "Large Model", "Filter Studio", "Gallery")
menu_buttons = {
    item_name: {
        "button_class": button(
            None,
            item_name,
            rounded=True,
            corner_radius=scale_ui(12),
            button_side_length=scale_ui(58),
            text_size=0.4 * SCALE_MIN,
            color=color,
            on_select=partial(select_menu_item, item_name),
        ),
        "button_sequence": index,
    }
    for index, (item_name, color) in enumerate(
        (
            ("Capture", (60, 100, 220)),
            ("Record", (70, 150, 70)),
            ("Recognize", (210, 130, 50)),
            ("OCR", (170, 80, 170)),
            ("amap api", (40, 170, 190)),
            ("Large Model", (190, 100, 60)),
            ("Filter Studio", (100, 180, 210)),
            ("Gallery", (80, 160, 220)),
        ),
        start=1,
    )
}
video_submenu_items = ("Pause", "Stop")
video_submenu_buttons = {
    item_name: {
        "button_class": button(
            None,
            item_name,
            rounded=True,
            corner_radius=scale_ui(10),
            button_side_length=scale_ui(68),
            text_size=0.4 * SCALE_MIN,
            color=color,
            on_select=partial(select_menu_item, item_name),
        ),
        "button_sequence": index,
    }
    for index, (item_name, color) in enumerate(
        (
            ("Pause", (120, 120, 120)),
            ("Stop", (180, 60, 60)),
        ),
        start=1,
    )
}
filter_submenu_items = ("default", "Sobel", "Scharr", "Sharpen", "Relief", "Old Photo", "Sketch")
filter_submenu_buttons = {
    item_name: {
        "button_class": button(
            None,
            item_name,
            rounded=True,
            corner_radius=scale_ui(8),
            button_side_length=scale_ui(50),
            text_size=0.38 * SCALE_MIN,
            color=color,
            on_select=partial(select_menu_item, item_name),
        ),
        "button_sequence": index,
    }
    for index, (item_name, color) in enumerate(
        (
            ("default", (120, 120, 120)),
            ("Sobel", (70, 120, 180)),
            ("Scharr", (120, 90, 180)),
            ("Sharpen", (80, 160, 120)),
            ("Relief", (180, 110, 80)),
            ("Old Photo", (170, 150, 100)),
            ("Sketch", (140, 140, 150)),
        ),
        start=1,
    )
}
recognition_submenu_items = tuple(get_model_names())
recognition_submenu_buttons = {
    item_name: {
        "button_class": button(
            None,
            item_name,
            rounded=True,
            corner_radius=scale_ui(8),
            button_side_length=scale_ui(52),
            text_size=0.36 * SCALE_MIN,
            color=color,
            on_select=partial(select_menu_item, item_name),
        ),
        "button_sequence": index,
    }
    for index, (item_name, color) in enumerate(
        (
            ("DEFAULT", (120, 120, 120)),
            ("Eye Detection Model", (70, 120, 180)),
            ("Eyeglass Eye Detection Model", (90, 130, 200)),
            ("Cat Face Detection Model", (100, 150, 100)),
            ("Enhanced Cat Face Detection Model", (120, 120, 100)),
            ("Face Detection Model (Focus on Side Face)", (140, 85, 120)),
            ("Face Detection Model (Version 2)", (110, 120, 180)),
            ("Face Detection Model (Tree Structure)", (150, 80, 100)),
            ("Face Detection Model (Default)", (120, 100, 160)),
            ("Full Body Detection Model", (70, 150, 180)),
            ("Left Eye Detection Model", (150, 120, 70)),
            ("Russian License Plate Detection Model", (165, 105, 90)),
            ("Lower Body Detection Model", (80, 120, 140)),
            ("Profile Face Detection Model", (110, 170, 100)),
            ("Right Eye Detection Model", (130, 130, 150)),
            ("Russian License Plate Detection Model (Backup)", (180, 140, 80)),
            ("Smile Detection Model", (140, 110, 190)),
            ("Upper Body Detection Model", (150, 160, 200)),
        ),
        start=1,
    )
}
menu = main_menu
menu_level = 1
MAX_MENU_LEVEL = 2
video_submenu_index = 0
filter_submenu_index = 0
recognition_submenu_index = 0
menu_context = "main"
current_model_name = "DEFAULT"
recognition_model = Model(current_model_name)


def get_active_recognition_status() -> tuple[str, tuple[int, int, int]]:
    """Return the current model label and the color used to draw it on the frame."""
    total = len(recognition_submenu_items)
    index = recognition_submenu_items.index(current_model_name) if current_model_name in recognition_submenu_items else 0
    if recognition_model.model_name == "DEFAULT":
        label = format_model_menu_label(current_model_name, index, total)
        return label, (0, 255, 0)
    if recognition_model.error_message:
        return recognition_model.error_message, (0, 0, 255)
    label = format_model_menu_label(recognition_model.model_name, index, total)
    return label, (0, 255, 0)


frame = np.zeros((480, 640, 3), dtype=np.uint8)
selected_index = 0
confirmed_index: int | None = None
last_frame_time = time.perf_counter()
fps = 0.0
stream = CameraStream(0)

while True:
    # 1. Read camera frames
    ret, frame = stream.read()
    if not ret:
        frame = stream._frame.copy()
        ret = stream.ret

    # 2. Handle keyboard events
    key_code = cv2.pollKey() if hasattr(cv2, "pollKey") else cv2.waitKey(1)
    key = key_code & 0xFF if key_code != -1 else -1
    if key == 27:
        break
    if key == ord("\t"):
        if menu_level >= MAX_MENU_LEVEL:
            menu_level = 1
        else:
            menu_level += 1
        menu_visible = True
        menu_last_action_time = time.time()
        continue
    if menu_level == 1:
        if key in (ord("w"),) or key_code in (2490368, 65362):
            selected_index = max(0, selected_index - 1)
            menu_visible = True
            menu_last_action_time = time.time()
        elif key in (ord("s"),) or key_code in (2621440, 65364):
            selected_index = min(len(menu_items) - 1, selected_index + 1)
            menu_visible = True
            menu_last_action_time = time.time()
        elif key in (13, 10):
            confirmed_index = selected_index
            selected_item = menu_items[confirmed_index]
            if selected_item == "Capture":
                photo_path = save_photo(frame)
                if photo_path is not None:
                    print(f"[Photo] saved: {photo_path}")
                else:
                    print("[Photo] save failed")
            elif selected_item == "Record":
                if recording_active:
                    stop_recording()
                else:
                    start_recording(frame)
                    menu_context = "video"
                    menu_level = 2
                    video_submenu_index = 0
            elif selected_item == "Filter Studio":
                menu_context = "filter"
                menu_level = 2
                filter_submenu_index = 0
            elif selected_item == "Recognize":
                menu_context = "recognition"
                menu_level = 2
                recognition_submenu_index = recognition_submenu_items.index(current_model_name) if current_model_name in recognition_submenu_items else 0
            elif selected_item == "Gallery":
                launch_gallery()
            else:
                menu_buttons[selected_item]["button_class"].select()
            menu_visible = True
            menu_last_action_time = time.time()
    else:
        if menu_context == "video":
            if key in (ord("w"),) or key_code in (2490368, 65362):
                video_submenu_index = max(0, video_submenu_index - 1)
                menu_visible = True
                menu_last_action_time = time.time()
            elif key in (ord("s"),) or key_code in (2621440, 65364):
                video_submenu_index = min(len(video_submenu_items) - 1, video_submenu_index + 1)
                menu_visible = True
                menu_last_action_time = time.time()
            elif key in (13, 10):
                selected_submenu = video_submenu_items[video_submenu_index]
                if selected_submenu == "Pause":
                    pause_recording()
                elif selected_submenu == "Stop":
                    stop_recording()
                    menu_context = "main"
                    menu_level = 1
                menu_visible = True
                menu_last_action_time = time.time()
        elif menu_context == "filter":
            if key in (ord("w"),) or key_code in (2490368, 65362):
                filter_submenu_index = max(0, filter_submenu_index - 1)
                menu_visible = True
                menu_last_action_time = time.time()
            elif key in (ord("s"),) or key_code in (2621440, 65364):
                filter_submenu_index = min(len(filter_submenu_items) - 1, filter_submenu_index + 1)
                menu_visible = True
                menu_last_action_time = time.time()
            elif key in (13, 10):
                selected_filter = filter_submenu_items[filter_submenu_index]
                current_filter = selected_filter
                menu_context = "main"
                menu_level = 1
                menu_visible = True
                menu_last_action_time = time.time()
        elif menu_context == "recognition":
            if key in (ord("w"),) or key_code in (2490368, 65362):
                recognition_submenu_index = max(0, recognition_submenu_index - 1)
                menu_visible = True
                menu_last_action_time = time.time()
            elif key in (ord("s"),) or key_code in (2621440, 65364):
                recognition_submenu_index = min(len(recognition_submenu_items) - 1, recognition_submenu_index + 1)
                menu_visible = True
                menu_last_action_time = time.time()
            elif key in (13, 10):
                selected_model_name = recognition_submenu_items[recognition_submenu_index]
                current_model_name = selected_model_name
                recognition_model = Model(selected_model_name)
                menu_context = "main"
                menu_level = 1
                menu_visible = True
                menu_last_action_time = time.time()
    if key == 32:
        if recording_active:
            stop_recording()
        ge = GaussianEffect(frame)
        frames = ge.generate_gaussian_effects((131, 131), 10, "progressive_blur", lighter=-100)
        for effect_frame in frames:
            cv2.imshow("0", effect_frame)
            cv2.waitKey(40)
        cv2.waitKey(0)

    # 3. Apply filters
    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
    if current_filter in {"Sobel", "Scharr", "Sharpen", "Relief", "Old Photo", "Sketch"}:
        frame = apply_filter_to_frame(frame, current_filter)

    # 3.5. Run recognition model when selected and draw result on the current frame
    if recognition_model.model_name != "DEFAULT" and recognition_model.classifier is not None:
        recognition_model.recognize(frame, display=True, color=(0, 255, 0), thickness=2)
        

    # 4. FPS calculation
    fps = stream.fps

    # 5. Add frame to video recorder
    if recording_active and not recording_paused and video_recorder is not None:
        camera_fps = fps
        if camera_fps > 0:
            video_recorder.sync_fps(camera_fps)
        video_recorder.add_frame(frame.copy())

    # 6. Render menu background (buttons handled separately)
    if menu_hide_seconds > 0 and menu_visible and time.time() - menu_last_action_time >= menu_hide_seconds:
        menu_visible = False
    if menu_visible:
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        if menu_level == 1:
            menu = main_menu
            active_buttons = menu_buttons
            selected_target = selected_index
            image = menu.render(frame, active_buttons)
            menu_origin = (
                int(menu.position[0] - menu.width / 2),
                int(menu.position[1] - menu.length / 2),
            )
            menu_region = image[
                menu_origin[1]:menu_origin[1] + int(menu.length),
                menu_origin[0]:menu_origin[0] + int(menu.width),
            ]
            t_menu_end = time.time()
            t_btn_start = time.time()
            rendered_buttons = render_buttons(menu, menu_region, active_buttons)
            if rendered_buttons:
                for index, rendered_button in enumerate(rendered_buttons):
                    button_position = rendered_button["button_position"]
                    canvas_position = (
                        button_position[0] + menu_origin[0],
                        button_position[1] + menu_origin[1],
                    )
                    if index == confirmed_index:
                        mark_button(image, canvas_position, "left", (0, 0, 255))
                    elif index == selected_index:
                        mark_button(image, canvas_position, "left", (255, 255, 255))
            t_btn_end = time.time()
        else:
            if menu_context == "video":
                menu = video_submenu
                active_buttons = video_submenu_buttons
                selected_target = video_submenu_index
                image = menu.render(frame, active_buttons)
                menu_origin = (
                    int(menu.position[0] - menu.width / 2),
                    int(menu.position[1] - menu.length / 2),
                )
                menu_region = image[
                    menu_origin[1]:menu_origin[1] + int(menu.length),
                    menu_origin[0]:menu_origin[0] + int(menu.width),
                ]
                t_menu_end = time.time()
                t_btn_start = time.time()
                rendered_buttons = render_buttons(menu, menu_region, active_buttons)
                if rendered_buttons:
                    for index, rendered_button in enumerate(rendered_buttons):
                        button_position = rendered_button["button_position"]
                        canvas_position = (
                            button_position[0] + menu_origin[0],
                            button_position[1] + menu_origin[1],
                        )
                        if index == video_submenu_index:
                            mark_button(image, canvas_position, "left", (255, 255, 255))
                t_btn_end = time.time()
            elif menu_context == "recognition":
                image = frame.copy()
                total_models = len(recognition_submenu_items)
                label_index = recognition_submenu_index if recognition_submenu_index < total_models else 0
                label_name = recognition_submenu_items[label_index]
                label = format_model_menu_label(label_name, label_index, total_models)
                label_x = int(main_menu.position[0] - scale_x(220))
                label_y = int(main_menu.position[1])
                cv2.putText(
                    image,
                    label,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 * SCALE_MIN,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                t_menu_end = time.time()
                t_btn_start = t_menu_end
                t_btn_end = t_menu_end
            else:
                menu = filter_submenu
                active_buttons = filter_submenu_buttons
                selected_target = filter_submenu_index
                image = menu.render(frame, active_buttons)
                menu_origin = (
                    int(menu.position[0] - menu.width / 2),
                    int(menu.position[1] - menu.length / 2),
                )
                menu_region = image[
                    menu_origin[1]:menu_origin[1] + int(menu.length),
                    menu_origin[0]:menu_origin[0] + int(menu.width),
                ]
                t_menu_end = time.time()
                t_btn_start = time.time()
                rendered_buttons = render_buttons(menu, menu_region, active_buttons)
                if rendered_buttons:
                    for index, rendered_button in enumerate(rendered_buttons):
                        button_position = rendered_button["button_position"]
                        canvas_position = (
                            button_position[0] + menu_origin[0],
                            button_position[1] + menu_origin[1],
                        )
                        if index == filter_submenu_index:
                            mark_button(image, canvas_position, "left", (255, 255, 255))
                t_btn_end = time.time()
    else:
        image = frame
        t_menu_end = time.time()
        t_btn_start = t_menu_end
        t_btn_end = t_menu_end

    # 8. Draw time and FPS info
    t_draw_start = time.time()
    # image=frame#if disable menu, use this line
    status_bar_y0, status_bar_h = 0, scale_y(30)
    cv2.rectangle(image, (0, status_bar_y0), (WINDOW_WIDTH, status_bar_h), (0, 0, 0), -1)
    time_text = time.strftime("%H:%M")
    font_scale = 0.7 * SCALE_MIN
    time_size, _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    time_x = scale_x(12)
    time_y = scale_y(22)
    fps_text = f"FPS: {fps:.1f}"
    fps_x = time_x + time_size[0] + scale_x(18)
    cv2.putText(image, time_text, (time_x, time_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, fps_text, (fps_x, time_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    if recording_active:
        record_elapsed = recording_elapsed_accumulated if recording_paused else time.time() - recording_start_time
        record_minutes, record_seconds = divmod(int(record_elapsed), 60)
        record_hours, record_minutes = divmod(record_minutes, 60)
        record_text = f"{record_hours:02d}:{record_minutes:02d}:{record_seconds:02d}"
        cv2.putText(image, record_text, (scale_x(640), scale_y(22)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2, cv2.LINE_AA)

    recognition_text, recognition_color = get_active_recognition_status()
    cv2.putText(
        image,
        recognition_text,
        (scale_x(10), scale_y(560)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55 * SCALE_MIN,
        recognition_color,
        2,
        cv2.LINE_AA,
    )
    bottom_font_scale = 0.65 * SCALE_MIN
    cv2.putText(image, f"Selected: {currentIem} | Filter: {current_filter}", (scale_x(10), scale_y(590)), cv2.FONT_HERSHEY_SIMPLEX, bottom_font_scale, (255, 0, 0), 2, cv2.LINE_AA)
    t_draw_end = time.time()
    result = image.get() if isinstance(image, cv2.UMat) else image
    cv2.imshow("0", result)

stream.release()
cv2.destroyAllWindows()