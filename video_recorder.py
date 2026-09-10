"""Video recording utility that keeps output FPS aligned with the camera frame rate."""

import os
import time
from pathlib import Path

import cv2
import numpy as np


class VideoRecorder:
    """Record OpenCV frames to disk with a configurable output size and codec.

    The class keeps a `fps` value synchronized with the live camera feed. When the
    source frame rate changes, call `sync_fps()` before `add_frame()` or pass the
    current camera FPS into the writer loop.
    """

    def __init__(
        self,
        save_path: str | os.PathLike[str],
        codec: str = "mp4v",
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
    ) -> None:
        """Initialize a recorder.

        Args:
            save_path: Target video file path, such as "recordings/cam1.mp4".
            codec: OpenCV FOURCC codec string, such as "mp4v", "MJPG", or "XVID".
            width: Output frame width in pixels.
            height: Output frame height in pixels.
            fps: Initial output FPS. This is updated by `sync_fps()` in real time.
        """
        self.save_path = str(save_path)
        self.codec = codec.upper()
        self.width = int(width)
        self.height = int(height)
        self.fps = max(float(fps), 1.0)
        self._writer: cv2.VideoWriter | None = None
        self._started = False
        self._last_frame_time = time.perf_counter()

    def sync_fps(self, current_fps: float) -> float:
        """Update the output FPS to match the current camera feed.

        Args:
            current_fps: Real-time FPS reported by the camera capture.

        Returns:
            The normalized output FPS after synchronization.
        """
        synced = max(float(current_fps), 1.0)
        self.fps = synced
        if self._writer is not None:
            self._writer.set(cv2.CAP_PROP_FPS, self.fps)
        return self.fps

    def _ensure_writer(self) -> None:
        """Create the OpenCV video writer if it has not been initialized yet."""
        directory = os.path.dirname(self.save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        if self._writer is not None and self._writer.isOpened():
            return

        if len(self.codec) != 4:
            if self.save_path.lower().endswith(".mp4"):
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            elif self.save_path.lower().endswith(".avi"):
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            else:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        else:
            fourcc = cv2.VideoWriter_fourcc(*self.codec)

        self._writer = cv2.VideoWriter(self.save_path, fourcc, self.fps, (self.width, self.height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.save_path}")
        self._started = True

    def add_frame(self, frame: np.ndarray) -> bool:
        """Append one camera frame to the video output.

        The frame is resized to the configured width and height before writing. The
        method also keeps the output frame rate aligned with the current camera FPS
        when `sync_fps()` has been called.

        Args:
            frame: An OpenCV BGR frame captured from the camera.

        Returns:
            True when the frame was written successfully; otherwise False.
        """
        if frame is None:
            return False

        resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self._ensure_writer()

        if self._writer is None:
            return False

        success = self._writer.write(resized)
        self._last_frame_time = time.perf_counter()
        return bool(success)

    def save(self) -> str:
        """Release the writer and finalize the recording.

        Returns:
            The saved file path.
        """
        if self._writer is not None:
            self._writer.release()
        return self.save_path


if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    recorder = VideoRecorder("recordings/camera_record.mp4", codec="mp4v", width=640, height=480, fps=30)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        current_fps = capture.get(cv2.CAP_PROP_FPS)
        if current_fps <= 0:
            current_fps = 30.0
        recorder.sync_fps(current_fps)
        recorder.add_frame(frame)

        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    recorder.save()
    capture.release()
    cv2.destroyAllWindows()
