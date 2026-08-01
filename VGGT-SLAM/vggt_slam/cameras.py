"""Camera backends for real-time SLAM."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Camera(ABC):
    """Abstract camera that produces BGR uint8 frames."""

    @abstractmethod
    def start(self) -> None:
        """Open the device and begin streaming."""
        ...

    @abstractmethod
    def capture(self) -> Optional[np.ndarray]:
        """Return a BGR uint8 (H, W, 3) frame, or None if unavailable."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Release the device."""
        ...


class RealSenseCamera(Camera):
    """Intel RealSense color stream via pyrealsense2."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self._width = width
        self._height = height
        self._fps = fps
        self._pipeline = None

    def start(self) -> None:
        import pyrealsense2 as rs

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)
        self._pipeline.start(config)

    def capture(self) -> Optional[np.ndarray]:
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        return np.asanyarray(color.get_data()) if color else None

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None


BACKENDS = {
    "realsense": RealSenseCamera,
}
