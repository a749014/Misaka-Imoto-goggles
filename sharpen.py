import cv2
import numpy as np


class Sharpen:
    """Image sharpening utility supporting multiple sharpening methods."""

    def __init__(self, src: np.ndarray, kernel_width: int = 2, kernel_height: int = 2):
        """
        Args:
            src: Input image (BGR or grayscale)
            kernel_width: Morphological kernel width
            kernel_height: Morphological kernel height
        """
        self.src = src
        self.kernel = np.ones((kernel_height, kernel_width), np.uint8)

    def sharpen(self, iterations: int = 1) -> np.ndarray:
        """Morphological sharpening based on erosion + dilation."""
        eroded = cv2.erode(self.src, self.kernel, iterations=iterations)
        return cv2.dilate(eroded, self.kernel, iterations=iterations)

    def unsharp_mask(self, blur_ksize: tuple = (5, 5), sigma: float = 1.0, amount: float = 1.5) -> np.ndarray:
        """
        Unsharp masking technique:
        blur the image first, subtract the blurred version from the original to get the detail layer,
        and then blend it back into the original image.
        """
        blurred = cv2.GaussianBlur(self.src, blur_ksize, sigma)
        sharpened = cv2.addWeighted(self.src, 1.0 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def laplacian(self, ksize: int = 3, scale: float = 1.0) -> np.ndarray:
        """
        Laplacian sharpening.
        Use the Laplacian operator to extract edges/high-frequency details and overlay them back onto the original.
        """
        lap = cv2.Laplacian(self.src, cv2.CV_64F, ksize=ksize)
        sharpened = self.src.astype(np.float64) - scale * lap
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def kernel_convolution(self, strength: float = 1.0) -> np.ndarray:
        """
        Convolution-based sharpening using a standard 3x3 kernel.
        The center weight is 1 + 4*strength and the neighbors are -strength.
        """
        kernel = np.array([
            [0, -strength, 0],
            [-strength, 1 + 4 * strength, -strength],
            [0, -strength, 0],
        ], dtype=np.float32)
        return cv2.filter2D(self.src, -1, kernel)


if __name__ == "__main__":
    img = cv2.imread("image.jpg")
    sharpener = Sharpen(img, kernel_width=2, kernel_height=2)

    morph = sharpener.sharpen()
    unsharp = sharpener.unsharp_mask()
    laplacian = sharpener.laplacian()
    conv = sharpener.kernel_convolution()

    cv2.imshow("Original", img)
    cv2.imshow("Morphological", morph)
    cv2.imshow("Unsharp Mask", unsharp)
    cv2.imshow("Laplacian", laplacian)
    cv2.imshow("Kernel Convolution", conv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
