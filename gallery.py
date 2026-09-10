"""Fullscreen OpenCV gallery for images and videos in the gallery directory."""

import os
import re
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

from image_warp import ImageWrapper


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".wmv"}


def get_current_resolution() -> Tuple[int, int]:
    """Read the primary display resolution from Linux XRandR.

    Args:
        None. The function uses the current X11 display selected by the
        ``DISPLAY`` environment variable.

    Returns:
        A ``(width, height)`` tuple in pixels for the first connected display.

    Raises:
        RuntimeError: If XRandR is unavailable, cannot query the display, or
            returns no connected display resolution.

    Example:
        width, height = get_current_resolution()
        print(f"Current resolution: {width}x{height}")
    """
    try:
        result = subprocess.run(
            ["xrandr", "--current"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Could not query the display with xrandr") from error

    for line in result.stdout.splitlines():
        if " connected" not in line:
            continue
        match = re.search(r"\b(\d+)x(\d+)\+\d+\+\d+\b", line)
        if match:
            return int(match.group(1)), int(match.group(2))
    raise RuntimeError("xrandr returned no connected display resolution")


class Gallery:
    """Browse media files in a newest-first fullscreen OpenCV window."""

    def __init__(self, settings_path: str = "settings.conf") -> None:
        """Initialize the gallery from a configuration file.

        Args:
            settings_path: Path to a key=value settings file. Missing files are
                created with defaults.

        Returns:
            None. The viewer is started by calling run().

        Example:
            gallery = Gallery("settings.conf")
            gallery.run()
        """
        self.settings = self._load_settings(settings_path)
        self.gallery_dir = self.settings["gallery_dir"]
        self.window_name = self.settings["window_name"]
        self.wrap_enabled = self.settings["wrap_enabled"]
        self.wrap_counts = self.settings["wrap_counts"]
        self.frame_delay_ms = self.settings["frame_delay_ms"]
        self.media = self._scan_media()
        self.index = 0
        self.canvas_size = (0, 0)
        self.show_change_time = False
        self.show_properties = False
        self.is_playing_video = False

    def _load_settings(self, settings_path: str) -> Dict[str, object]:
        """Read viewer settings and create the file when it is absent.

        Args:
            settings_path: Path to a UTF-8 key=value configuration file.

        Returns:
            A dictionary containing validated gallery, window, and transition settings.

        Example:
            settings = gallery._load_settings("settings.conf")
        """
        defaults: Dict[str, object] = {
            "gallery_dir": "gallery",
            "window_name": "Gallery",
            "wrap_enabled": True,
            "wrap_counts": 12,
            "frame_delay_ms": 25,
        }
        if not os.path.exists(settings_path):
            with open(settings_path, "w", encoding="utf-8") as config_file:
                config_file.write(
                    "gallery_dir=gallery\n"
                    "window_name=Gallery\n"
                    "wrap_enabled=true\n"
                    "wrap_counts=12\n"
                    "frame_delay_ms=25\n"
                )

        values = dict(defaults)
        with open(settings_path, "r", encoding="utf-8") as config_file:
            for line in config_file:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = (part.strip() for part in line.split("=", 1))
                if key == "gallery_dir":
                    values[key] = value or "gallery"
                elif key == "window_name":
                    values[key] = value or "Gallery"
                elif key == "wrap_enabled":
                    values[key] = value.lower() in {"1", "true", "yes", "on"}
                elif key in {"wrap_counts", "frame_delay_ms"}:
                    try:
                        parsed = int(value)
                        if parsed > 0:
                            values[key] = parsed
                    except ValueError:
                        pass

        os.makedirs(str(values["gallery_dir"]), exist_ok=True)
        return values

    def _scan_media(self) -> Dict[str, float]:
        """Find supported media and map filenames to modification timestamps.

        Args:
            None. The configured gallery directory is scanned.

        Returns:
            An insertion-ordered dictionary whose keys are filenames and whose
            values are modification timestamps, newest first.

        Example:
            files = gallery._scan_media()
        """
        media: List[Tuple[float, str]] = []
        for name in os.listdir(self.gallery_dir):
            path = os.path.join(self.gallery_dir, name)
            extension = os.path.splitext(name)[1].lower()
            if os.path.isfile(path) and extension in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
                media.append((os.path.getmtime(path), path))
        media.sort(key=lambda item: item[0], reverse=True)
        return {os.path.basename(path): changed_at for changed_at, path in media}

    def _media_path(self, filename: str) -> str:
        """Build the path for a filename stored in the media dictionary.

        Args:
            filename: Media filename returned by _scan_media().

        Returns:
            The path to the file inside the configured gallery directory.

        Example:
            path = gallery._media_path("photo.jpg")
        """
        return os.path.join(self.gallery_dir, filename)

    def _draw_change_time(self, frame, filename: str):
        """Draw a media file's modification time at the top-left when enabled.

        Args:
            frame: BGR frame on which the text should be drawn.
            filename: Key whose timestamp is read from self.media.

        Returns:
            The frame with an optional timestamp overlay.

        Example:
            frame = gallery._draw_change_time(frame, "photo.jpg")
        """
        # print(f"show_change_time: {self.show_change_time}, filename: {filename}, media: {self.media}")
        if not self.show_change_time or filename not in self.media:
            return frame
        changed_at = datetime.fromtimestamp(self.media[filename]).strftime("%Y-%m-%d %H:%M:%S")
        # print(f"Drawing timestamp for {filename}: {changed_at}")
        cv2.putText(
            frame,
            f"Changed: {changed_at}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def _draw_centered_lines(self, frame, lines: List[str], color: Tuple[int, int, int]):
        """Draw multiple text lines centered on a frame.

        Args:
            frame: BGR frame on which the lines are drawn.
            lines: Text lines to draw from top to bottom.
            color: BGR text color.

        Returns:
            The frame with the centered text lines.

        Example:
            frame = gallery._draw_centered_lines(frame, ["Delete? Y/N"], (0, 0, 255))
        """
        if not lines:
            return frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.9
        thickness = 2
        line_height = 40
        padding = 8 #
        total_height = len(lines) * line_height
        first_y = max(40, (frame.shape[0] - total_height) // 2)
        for line_number, text in enumerate(lines):
            text_size, text_baseline = cv2.getTextSize(text, font, scale, thickness)
            x = max(10, (frame.shape[1] - text_size[0]) // 2)
            y = first_y + line_number * line_height
            cv2.rectangle(
                frame,
                (x - padding, y - text_size[1] - padding),
                (x + text_size[0] + padding, y + text_baseline + padding),
                (0, 0, 0),
                cv2.FILLED,
            )
            cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        return frame

    def _media_properties(self, filename: str, frame) -> List[str]:
        """Build display lines describing the selected image or video.

        Args:
            filename: Filename whose properties should be displayed.
            frame: Current BGR frame used to report displayed dimensions.

        Returns:
            Property strings for centered, line-by-line rendering.

        Example:
            lines = gallery._media_properties("photo.jpg", frame)
        """
        path = self._media_path(filename)
        stat = os.stat(path)
        media_type = "Video" if self._is_video(filename) else "Image"
        lines = [
            f"Name: {filename}",
            f"Type: {media_type}",
            f"Size: {stat.st_size} bytes",
            f"Modified: {datetime.fromtimestamp(stat.st_mtime):%Y-%m-%d %H:%M:%S}",
            f"Resolution: {frame.shape[1]} x {frame.shape[0]}",
        ]
        if self._is_video(filename):
            capture = cv2.VideoCapture(path)
            fps = capture.get(cv2.CAP_PROP_FPS)
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            capture.release()
            duration = frame_count / fps if fps > 0 else 0.0
            lines.append(f"Duration: {duration:.1f} seconds")
        return lines

    def _delete_current(self, filename: str) -> bool:
        """Delete the selected media file after the caller confirms it.

        Args:
            filename: Filename of the current media item.

        Returns:
            True when the file was deleted, otherwise False after an error.

        Example:
            deleted = gallery._delete_current("photo.jpg")
        """
        try:
            os.remove(self._media_path(filename))
        except OSError as error:
            print(f"[Gallery] Could not delete {filename}: {error}")
            return False
        print(f"[Gallery] Deleted {filename}")
        return True

    def _confirm_delete(self, frame, filename: str) -> bool:
        """Ask for Y/N confirmation before deleting the selected media.

        Args:
            frame: Current BGR frame used as the confirmation background.
            filename: Filename shown in the confirmation prompt.

        Returns:
            True only when the user presses Y and deletion succeeds; N cancels.

        Example:
            deleted = gallery._confirm_delete(frame, "photo.jpg")
        """
        while True:
            prompt = frame.copy()
            self._draw_centered_lines(
                prompt,
                [f"Delete {filename}?", "Press Y to delete, N to cancel"],
                (0, 0, 255),
            )
            cv2.imshow(self.window_name, prompt)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("y"):
                return self._delete_current(filename)
            if key == ord("n"):
                return False

    def _is_video(self, path: str) -> bool:
        """Return whether a path has a supported video extension.

        Args:
            path: Media path to classify.

        Returns:
            True for video extensions and False for image extensions or unknown files.

        Example:
            is_movie = gallery._is_video("gallery/movie.mp4")
        """
        return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS

    def _fit_to_canvas(self, frame, width: int, height: int):
        """Resize a frame to exactly fill the current display resolution.

        Args:
            frame: BGR image frame returned by OpenCV.
            width: Target canvas width in pixels.
            height: Target canvas height in pixels.

        Returns:
            A BGR frame exactly sized to the target canvas. The full source
            frame remains visible, but it may be stretched when aspect ratios differ.

        Example:
            canvas = gallery._fit_to_canvas(frame, 1920, 1080)
        """
        if frame is None or frame.size == 0:
            return None
        horizontal_scale = width / frame.shape[1]
        vertical_scale = height / frame.shape[0]
        if horizontal_scale < 1.0 or vertical_scale < 1.0:
            # Area resampling averages source pixels, reducing aliasing when shrinking.
            interpolation = cv2.INTER_AREA
        elif horizontal_scale > 1.0 or vertical_scale > 1.0:
            # Lanczos preserves more edge detail than linear interpolation when enlarging.
            interpolation = cv2.INTER_LANCZOS4
        else:
            interpolation = cv2.INTER_NEAREST
        return cv2.resize(frame, (width, height), interpolation=interpolation)

    def _read_first_frame(self, path: str):
        """Load the first displayable frame from an image or video.

        Args:
            path: Supported image or video path.

        Returns:
            The first BGR frame, or None when OpenCV cannot read the media.

        Example:
            frame = gallery._read_first_frame("gallery/photo.jpg")
        """
        if self._is_video(path):
            capture = cv2.VideoCapture(path)
            success, frame = capture.read()
            capture.release()
            return frame if success else None
        return cv2.imread(path, cv2.IMREAD_COLOR)

    def _show_error_frame(self, message: str = "Error reading media. Press any key to continue.") -> int:
        """Display an error overlay on a blank frame and wait for user input.

        Args:
            message: Message to render in red on a black canvas.

        Returns:
            The key code pressed by the user.

        Example:
            key = gallery._show_error_frame()
        """
        empty_frame = np.zeros(
            (max(1, self.canvas_size[1]), max(1, self.canvas_size[0]), 3),
            dtype=np.uint8,
        )
        cv2.putText(
            empty_frame,
            message,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(self.window_name, empty_frame)
        return cv2.waitKey(0) & 0xFF

    def _animate_to(self, old_frame, new_frame, direction: str) -> None:
        """Show synchronized page-turn animations for two media frames.

        Args:
            old_frame: Current BGR frame matching the display canvas size.
            new_frame: Next BGR frame matching the display canvas size.
            direction: "left" or "right", passed to ImageWrapper.

        Returns:
            None. Both warped frames are rendered together in the gallery window.

        Example:
            gallery._animate_to(current_frame, next_frame, "left")
        """
        if not self.wrap_enabled or self.canvas_size == (0, 0):
            cv2.imshow(self.window_name, new_frame)
            return
        height, width = old_frame.shape[:2]
        old_wrapper = ImageWrapper(old_frame, self.wrap_counts, width, height)
        new_wrapper = ImageWrapper(new_frame, self.wrap_counts, width, height)
        old_frames = old_wrapper.generate_img(old_wrapper.generate_Ms(direction))
        opposite_direction = "right" if direction == "left" else "left"
        new_frames = new_wrapper.generate_img(new_wrapper.generate_Ms(opposite_direction))
        new_frames.reverse()
        for index, (old_transition, new_transition) in enumerate(zip(old_frames, new_frames), start=1):
            progress = index / len(old_frames)
            transition = cv2.addWeighted(old_transition, 1.0 - progress,
                                         new_transition, progress, 0.0)
            cv2.imshow(self.window_name, transition)
            key = cv2.waitKey(self.frame_delay_ms) & 0xFF
            if key in (27, ord("q"), 81, 83):
                break

    def _show_image(self, filename: str) -> Optional[int]:
        """Display one image and wait for navigation or quit input.

        Args:
            filename: Image filename stored in self.media.

        Returns:
            The pressed key code, or -1 when the image cannot be loaded.

        Example:
            key = gallery._show_image("gallery/photo.jpg")
        """
        frame = self._read_first_frame(self._media_path(filename))
        if frame is None:
            print(f"[Gallery] Could not read {filename}")
            return -1
        frame = self._fit_to_canvas(frame, *self.canvas_size)
        while True:
            display_frame = self._draw_change_time(frame.copy(), filename)
            if self.show_properties:
                self._draw_centered_lines(
                    display_frame,
                    self._media_properties(filename, frame),
                    (255, 255, 255),
                )
            cv2.imshow(self.window_name, display_frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("d"):
                if self._confirm_delete(frame, filename):
                    return ord("d")
                continue
            if key == ord("p"):
                self.show_properties = not self.show_properties
                continue
            if key != ord("t"):
                return key
            self.show_change_time = not self.show_change_time

    def _show_video(self, filename: str) -> int:
        """Play a video until it ends or the user presses a navigation key.

        Args:
            filename: Video filename stored in self.media.

        Returns:
            The pressed key code, or -1 when the video cannot be opened.

        Example:
            key = gallery._show_video("gallery/movie.mp4")
        """
        capture = cv2.VideoCapture(self._media_path(filename))
        if not capture.isOpened():
            print(f"[Gallery] Could not open {filename}")
            return self._show_error_frame()
        key = -1
        success, frame = capture.read()
        if not success:
            capture.release()
            print(f"[Gallery] Could not read the first frame of {filename}")
            return self._show_error_frame()
        while True:
            if self.is_playing_video:
                success, frame = capture.read()
                if not success:
                    break
            display_frame = self._fit_to_canvas(frame.copy(), *self.canvas_size)
            if not self.is_playing_video:
                cv2.putText(
                    display_frame,
                    "Paused: Space Play, P Properties, D Delete, T Time, Arrows Navigate, Q Quit",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            display_frame = self._draw_change_time(display_frame, filename)
            if self.show_properties:
                self._draw_centered_lines(
                    display_frame,
                    self._media_properties(filename, frame),
                    (255, 255, 255),
                )
            cv2.imshow(self.window_name, display_frame)
            key = cv2.waitKey(self.frame_delay_ms) & 0xFF
            if key == ord(" "):
                self.is_playing_video = not self.is_playing_video
                continue
            if key == ord("d"):
                self.is_playing_video = False
                if self._confirm_delete(display_frame, filename):
                    capture.release()
                    return ord("d")
                continue
            if key == ord("p"):
                self.show_properties = not self.show_properties
                continue
            if key == ord("t"):
                self.show_change_time = not self.show_change_time
                print("Finished: toggle timestamp", self.show_change_time)
                continue
            if key in (27, ord("q"), 81, 83):
                break
        capture.release()
        return key

    def _move(self, step: int) -> bool:
        """Move by one media item and display its first frame with a transition.

        Args:
            step: Navigation increment, normally -1 for left or 1 for right.

        Returns:
            True when the index changed, or False at a non-wrapping boundary.

        Example:
            moved = gallery._move(1)
        """
        filenames = list(self.media)
        new_index = self.index + step
        if not 0 <= new_index < len(filenames):
            return False
        previous = self._read_first_frame(self._media_path(filenames[self.index]))
        target = self._read_first_frame(self._media_path(filenames[new_index]))
        self.index = new_index
        if target is not None:
            target = self._fit_to_canvas(target, *self.canvas_size)
            if previous is not None:
                self._animate_to(self._fit_to_canvas(previous, *self.canvas_size), target,
                                 "left" if step > 0 else "right")
            cv2.imshow(self.window_name, target)
        return True

    def run(self) -> None:
        """Open the fullscreen viewer and respond to left, right, and quit keys.

        Args:
            None. Media and behavior come from the configured gallery directory.

        Returns:
            None after Escape, q, or a closed window.

        Example:
            Gallery().run()
        """
        if not self.media:
            print(f"[Gallery] No supported media found in {self.gallery_dir}")
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        try:
            width, height = get_current_resolution()
        except RuntimeError as error:
            print(f"[Gallery] {error}; using OpenCV window size")
            _, _, width, height = cv2.getWindowImageRect(self.window_name)
        if width <= 1 or height <= 1:
            first_filename = next(iter(self.media))
            first_frame = self._read_first_frame(self._media_path(first_filename))
            if first_frame is not None:
                height, width = first_frame.shape[:2]
        self.canvas_size = (max(1, width), max(1, height))
        empty_frame = np.zeros((width,height, 3), dtype=np.uint8)
        print(empty_frame.shape)
        while True:
            filenames = list(self.media)
            filename = filenames[self.index]
            key = self._show_video(filename) if self._is_video(filename) else self._show_image(filename)
            # if key in (27, ord("q")):
            #     break
            if key == ord("d"):
                self.media = self._scan_media()
                if not self.media:
                    break
                self.index = min(self.index, len(self.media) - 1)
                continue
            if key == -1:
                key = self._show_error_frame()
            if key in (27, ord("q")):
                break
            if key == 81:
                self._move(-1)
            elif key == 83:
                self._move(1)
        cv2.destroyAllWindows()


def main() -> None:
    """Start the gallery using settings.conf in the current directory.

    Args:
        None. Configuration is loaded from the default settings.conf path.

    Returns:
        None after the gallery window closes.

    Example:
        main()
    """
    Gallery().run()


if __name__ == "__main__":
    main()
