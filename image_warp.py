import cv2
import numpy as np
import os
from typing import List
class ImageWrapper:
    """
    Perspective-based page-flip (page-turning) effect generator.

    Given a source image, this class produces a sequence of 3x3 perspective
    matrices and the corresponding warped frames. Playing the frames in order
    simulates a page being turned to the left or to the right.

    Usage example:
        >>> img = cv2.imread("page.jpg")
        >>> wrapper = ImageWrapper(src=img, wrap_counts=10, w=640, h=480)
        >>> Ms = wrapper.generate_Ms(direction="left")   # or "right"
        >>> frames = wrapper.generate_img(Ms)
        >>> for frame in frames:
        ...     cv2.imshow("flip", frame)
        ...     cv2.waitKey(100)
        >>> cv2.destroyAllWindows()
    """

    def __init__(self, src: np.ndarray, wrap_counts: int, w: int, h: int) -> None:
        """
        Initialize the wrapper with a source image and animation settings.

        Args:
            src (np.ndarray): Source image (BGR or grayscale), as returned by cv2.imread.
            wrap_counts (int): Total number of flip steps (animation frames). Must be >= 1.
            w (int): Width of the output frames, in pixels. Must be > 0.
            h (int): Height of the output frames, in pixels. Must be > 0.

        Raises:
            TypeError: If any argument has the wrong type.
            ValueError: If a numeric argument is out of range.

        Example:
            >>> wrapper = ImageWrapper(src=cv2.imread("page.jpg"), wrap_counts=10, w=640, h=480)
        """
        # ---- fail fast on bad types / values, instead of producing weird frames ----
        if not isinstance(src, np.ndarray):
            raise TypeError(f"src must be np.ndarray, got {type(src).__name__}")
        if not isinstance(wrap_counts, int) or isinstance(wrap_counts, bool):
            raise TypeError(f"wrap_counts must be int, got {type(wrap_counts).__name__}")
        if wrap_counts < 1:
            raise ValueError(f"wrap_counts must be >= 1, got {wrap_counts}")
        if not isinstance(w, int) or not isinstance(h, int) or isinstance(w, bool) or isinstance(h, bool):
            raise TypeError("w and h must be int")
        if w <= 0 or h <= 0:
            raise ValueError("w and h must be positive")
        # Soft warning: it still works, but usually src should match (w, h).
        if src.ndim < 2 or src.shape[:2] != (h, w):
            print(f"[ImageWrapper] Warning: src shape {src.shape[:2]} != (h, w) = ({h}, {w}), "
                  f"output may not look as expected.")

        self.src = src
        self.wrap_counts = wrap_counts
        self.w = w
        self.h = h

    @staticmethod
    def _is_valid_M(M: object) -> bool:
        """
        Check whether an object is a valid perspective matrix.

        A valid matrix (as returned by cv2.getPerspectiveTransform) is an
        np.ndarray of shape (3, 3) with a floating dtype.
        """
        return (
            isinstance(M, np.ndarray)
            and M.shape == (3, 3)
            and np.issubdtype(M.dtype, np.floating)
        )

    def generate_Ms(self, direction: str = "left") -> List[np.ndarray]:
        """
        Generate the sequence of perspective matrices for one flip animation.

        The source rectangle is always the full image (pixel coordinates):
            src = [[0,0], [w,0], [0,h], [w,h]]
                   top-left, top-right, bottom-left, bottom-right

        - "left":  the LEFT edge is the fixed spine; the right edge moves
                   leftwards while its corners shear towards the vertical middle.
        - "right": the RIGHT edge is the fixed spine; the left edge moves
                   rightwards while its corners shear towards the vertical middle.

        Args:
            direction (str): "left" or "right" (case-insensitive).
                Any other value raises ValueError.

        Returns:
            List[np.ndarray]: wrap_counts matrices of shape (3, 3) (dtype float64),
            ordered from the first animation step to the last.

        Raises:
            ValueError: If direction is not "left" or "right".

        Example:
            >>> Ms_left = wrapper.generate_Ms("left")
            >>> Ms_right = wrapper.generate_Ms("right")
        """
        if not isinstance(direction, str) or direction.lower() not in ("left", "right"):
            raise ValueError(f"direction must be 'left' or 'right', got {direction!r}")
        direction = direction.lower()

        # Full-image source rectangle in pixel coordinates.
        src = np.array([[0, 0], [self.w, 0], [0, self.h], [self.w, self.h]], np.float32)

        Ms: List[np.ndarray] = []
        for i in range(1, self.wrap_counts + 1):
            t = i / self.wrap_counts  # normalized flip progress, in (0, 1]

            if direction == "left":
                # Left flip: LEFT edge fixed (the spine), RIGHT edge folds in.
                dst = np.array(
                    [
                        [0, 0],                                      # top-left     (fixed)
                        [self.w * (1 - t), self.h * 0.5 * t],        # top-right    -> left + down
                        [0, self.h],                                 # bottom-left  (fixed)
                        [self.w * (1 - t), self.h * (1 - 0.5 * t)],  # bottom-right -> left + up
                    ],
                    np.float32,
                )
            else:
                # Right flip: RIGHT edge fixed (the spine), LEFT edge folds in.
                # Mirrored version of the left-flip dst across the vertical center line.
                dst = np.array(
                    [
                        [self.w * t, self.h * 0.5 * t],              # top-left     -> right + down
                        [self.w, 0],                                 # top-right    (fixed)
                        [self.w * t, self.h * (1 - 0.5 * t)],        # bottom-left  -> right + up
                        [self.w, self.h],                            # bottom-right (fixed)
                    ],
                    np.float32,
                )

            Ms.append(cv2.getPerspectiveTransform(src, dst))
        return Ms

    def generate_img(self, Ms: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply every perspective matrix to the source image.

        Each valid matrix is checked with _is_valid_M() BEFORE being used:
        invalid entries (wrong type / shape / dtype) are skipped with a
        warning instead of crashing the whole animation.

        Args:
            Ms (List[np.ndarray]): Perspective matrices,
                e.g. the return value of generate_Ms().

        Returns:
            List[np.ndarray]: Warped frames, all of size (h, w); the k-th frame
            corresponds to Ms[k]. The list may be shorter than Ms if some
            entries were invalid.

        Example:
            >>> frames = wrapper.generate_img(wrapper.generate_Ms("left"))
        """
        if not isinstance(Ms, (list, tuple)):
            raise TypeError(f"Ms must be a list of 3x3 matrices, got {type(Ms).__name__}")

        imgs: List[np.ndarray] = []
        for idx, M in enumerate(Ms):
            if not self._is_valid_M(M):
                print(f"[ImageWrapper] Warning: Ms[{idx}] is not a valid 3x3 float "
                      f"perspective matrix (got {type(M).__name__}), skipped.")
                continue
            imgs.append(cv2.warpPerspective(self.src, M, (self.w, self.h)))
        return imgs
if __name__ == "__main__":
    img = cv2.imread("./image.jpg")
    if img is None:
        raise FileNotFoundError("./image.jpg could not be loaded")
    wrapper = ImageWrapper(img, 3, len(img[0]), len(img))
    new_imgs = wrapper.generate_img(wrapper.generate_Ms())
    for index, frame in enumerate(new_imgs):
        cv2.imshow(f"flip-{index}", frame)
        cv2.waitKey(100)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
