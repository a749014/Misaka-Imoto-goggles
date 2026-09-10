import cv2
import numpy as np
from typing import Optional, Union


class MultipleFilters:

        def __init__(self, image, resize_factor: float = 1) -> None:
                """
                Initialize the detector by loading an image and optionally resizing it.

                Args:
                        image_path (str): Path to the source image file.
                        resize_factor (float): Resizing scale, e.g. 0.5 means half size.

                Example:
                        detector = EdgeDetector("./image2.png", resize_factor=0.5)
                """
                if resize_factor <= 0:
                        raise ValueError("resize_factor must be greater than zero")

                image = image if isinstance(image, np.ndarray) else cv2.imread(image)
                if image is None:
                        raise FileNotFoundError(f"Could not load image: ")
                if resize_factor != 1.0:
                        image = cv2.resize(image, None, fx=resize_factor, fy=resize_factor)
                self.image: np.ndarray = image

        def _resolve_image(self, image: Optional[np.ndarray] = None) -> np.ndarray:
                """
                Return the provided image or the default image stored in self.image.

                Args:
                        image (Optional[np.ndarray]): Optional input image array.
                                If None, use self.image.

                Returns:
                        np.ndarray: The image used for processing.
                """
                if image is None:
                        if not hasattr(self, "image") or self.image is None:
                                raise ValueError("No image is available in EdgeDetector.")
                        return self.image
                else:
                        if not isinstance(image, np.ndarray):
                                raise TypeError("Provided image must be a numpy ndarray")
                        return image

        def canny_detector(
                self,
                image: Optional[np.ndarray] = None,
                lower_threshold: int = 120,
                upper_threshold: int = 200,
                l2_gradient: bool = True,
                is_blur: bool = False,
        ) -> np.ndarray:
                """
                Detect object edges using the Canny edge detector.

                Args:
                        image (Optional[np.ndarray]): Optional image array to process.
                                If None, use self.image.
                        lower_threshold (int): Lower hysteresis threshold in [0, 255].
                                Smaller values detect more weak edges.
                        upper_threshold (int): Upper hysteresis threshold in [0, 255].
                                Larger values suppress weak edges.
                        l2_gradient (bool): Whether to use L2 norm for gradient magnitude.
                                True gives more stable edge detection.
                        is_blur (bool): If True, apply Gaussian blur before Canny detection.

                Returns:
                        np.ndarray: Edge map image in binary form.

                Example:
                        edges = detector.canny_detector(
                                lower_threshold=120,
                                upper_threshold=200,
                                l2_gradient=True,
                                is_blur=True,
                        )
                """
                if not isinstance(lower_threshold, int) or not isinstance(upper_threshold, int):
                        raise TypeError("Canny thresholds must be integers")
                if not 0 <= lower_threshold <= upper_threshold <= 255:
                        raise ValueError(
                                "thresholds must satisfy 0 <= lower_threshold <= upper_threshold <= 255"
                        )

                source = self._resolve_image(image)
                if source.ndim == 3:
                        source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

                if is_blur:
                        source = cv2.GaussianBlur(source, (3, 3), 0)

                return cv2.Canny(
                        source,
                        lower_threshold,
                        upper_threshold,
                        L2gradient=l2_gradient,
                )

        def counter_extract(
                self,
                image: Optional[np.ndarray] = None,
                return_mode: str = "contours",
                color: tuple[int, int, int] = (255, 0, 0),
                threshold: int = 127,
                contour_retrieval_mode: int = cv2.RETR_EXTERNAL,
                contour_approximation_method: int = cv2.CHAIN_APPROX_NONE,
                thickness: int = 2,
        ) -> Union[list[np.ndarray], np.ndarray]:
                """
                Extract contours from a binary image and return either the contours list
                or the original image with contours drawn on it.

                Args:
                        image (Optional[np.ndarray]): Optional image to process.
                                If None, use self.image.
                        return_mode (str): How the function returns data.
                                "contours" returns a list of contour arrays.
                                "image" returns an image with drawn contours.
                        color (tuple[int, int, int]): **BGR** color for contour drawing,
                                such as (255, 0, 0) for blue.
                        threshold (int): Binary threshold value in [0, 255].
                                Pixels above threshold become white in the binary image.
                        contour_retrieval_mode (int): OpenCV contour retrieval mode,
                                such as cv2.RETR_EXTERNAL or cv2.RETR_TREE.
                        contour_approximation_method (int): OpenCV contour approximation mode,
                                such as cv2.CHAIN_APPROX_NONE or cv2.CHAIN_APPROX_SIMPLE.
                        thickness (int): Line thickness for drawing contours.
                                Use 1 or 2 for common output.

                Returns:
                        Union[list[np.ndarray], np.ndarray]: Either a list of contours or
                        a modified image array with contours painted.

                Example:
                        contours = detector.counter_extract(
                                return_mode="contours",
                                threshold=127,
                                contour_retrieval_mode=cv2.RETR_EXTERNAL,
                                contour_approximation_method=cv2.CHAIN_APPROX_NONE,
                        )

                        image_with_contours = detector.counter_extract(
                                return_mode="image",
                                color=(255, 0, 0),
                                threshold=127,
                                thickness=2,
                        )
                """
                if return_mode not in {"contours", "image"}:
                        raise ValueError("return_mode must be 'contours' or 'image'")
                if not isinstance(threshold, int):
                        raise TypeError("threshold must be an integer")
                if not 0 <= threshold <= 255:
                        raise ValueError("threshold must satisfy 0 <= threshold <= 255")
                if not isinstance(color, tuple) or len(color) != 3:
                        raise TypeError("color must be a tuple of three integers in BGR format")
                if any(not isinstance(channel, int) for channel in color):
                        raise TypeError("each color channel must be an integer in [0, 255]")
                if any(not 0 <= channel <= 255 for channel in color):
                        raise ValueError("each color channel must be in the range [0, 255]")
                if not isinstance(thickness, int):
                        raise TypeError("thickness must be an integer")
                if thickness <= 0:
                        raise ValueError("thickness must be greater than zero")

                draw_target = self._resolve_image(image)
                

                if draw_target.ndim == 3:
                        print("color image detected, converting to grayscale for contour extraction")
                        gray_image = cv2.cvtColor(draw_target, cv2.COLOR_BGR2GRAY)
                else:
                        gray_image = draw_target

                _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(
                        binary_image,
                        contour_retrieval_mode,
                        contour_approximation_method,
                )

                if return_mode == "contours":
                        return contours

                return cv2.drawContours(draw_target, contours, -1, color, thickness)

        def sobel_operator(
                self,
                image: Optional[np.ndarray] = None,
                dx: int = 1,
                dy: int = 0,
                ksize: int = 3,
                x_weight: float = 1.0,
                y_weight: float = 1.0,
                scale: float = 1.0,
                delta: float = 0,
        ) -> np.ndarray:
                """
                Apply Sobel edge detection to a source image and combine the x/y responses
                by a configurable weight ratio.

                Args:
                        image (Optional[np.ndarray]): Image to process. If omitted, use self.image.
                        dx (int): X-order derivative. 0 means only vertical gradients.
                        dy (int): Y-order derivative. 0 means only horizontal gradients.
                        ksize (int): Sobel kernel size, usually 1, 3, 5, or 7.
                        x_weight (float): Weight applied to the Sobel x response.
                        y_weight (float): Weight applied to the Sobel y response.
                        scale (float): Optional scaling factor before the output is converted.
                        delta (float): Optional value added to the result before conversion.

                Returns:
                        np.ndarray: Edge image with the combined Sobel response.
                """
                if dx < 0 or dy < 0:
                        raise ValueError("dx and dy must be non-negative integers")
                if dx == 0 and dy == 0:
                        raise ValueError("dx and dy cannot both be zero")
                if ksize not in {1, 3, 5, 7}:
                        raise ValueError("ksize must be one of 1, 3, 5, or 7")
                if x_weight <= 0 or y_weight <= 0:
                        raise ValueError("x_weight and y_weight must be greater than zero")

                source = self._resolve_image(image)
                gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if source.ndim == 3 else source

                sobel_x = np.zeros_like(gray, dtype=np.float64)
                sobel_y = np.zeros_like(gray, dtype=np.float64)

                if dx > 0:
                        sobel_x = cv2.Sobel(gray, cv2.CV_64F, dx, 0, ksize=ksize, scale=scale, delta=delta)
                if dy > 0:
                        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, dy, ksize=ksize, scale=scale, delta=delta)

                total_weight = x_weight + y_weight
                weight_x = x_weight / total_weight
                weight_y = y_weight / total_weight

                sobel_x_abs = cv2.convertScaleAbs(sobel_x)
                sobel_y_abs = cv2.convertScaleAbs(sobel_y)
                combined = cv2.addWeighted(sobel_x_abs, weight_x, sobel_y_abs, weight_y, 0)
                return combined

        def scharr_operator(
                self,
                image: Optional[np.ndarray] = None,
                dx: int = 1,
                dy: int = 0,
                x_weight: float = 1.0,
                y_weight: float = 1.0,
                scale: float = 1.0,
                delta: float = 0,
        ) -> np.ndarray:
                """
                Apply Scharr edge detection to a source image and mix the x/y responses
                using a user-defined weight ratio.

                Args:
                        image (Optional[np.ndarray]): Image to process. If omitted, use self.image.
                        dx (int): X gradient order, 0 or 1.
                        dy (int): Y gradient order, 0 or 1.
                        x_weight (float): Weight assigned to the Scharr x response.
                        y_weight (float): Weight assigned to the Scharr y response.
                        scale (float): Optional scale applied before conversion.
                        delta (float): Optional value added before conversion.

                Returns:
                        np.ndarray: Edge image with the combined Scharr response.
                """
                if dx not in {0, 1} or dy not in {0, 1}:
                        raise ValueError("dx and dy must be either 0 or 1 for Scharr operators")
                if dx == 0 and dy == 0:
                        raise ValueError("dx and dy cannot both be zero")
                if x_weight <= 0 or y_weight <= 0:
                        raise ValueError("x_weight and y_weight must be greater than zero")

                source = self._resolve_image(image)
                gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if source.ndim == 3 else source

                scharr_x = np.zeros_like(gray, dtype=np.float64)
                scharr_y = np.zeros_like(gray, dtype=np.float64)

                if dx > 0:
                        scharr_x = cv2.Scharr(gray, cv2.CV_64F, dx, 0, scale=scale, delta=delta)
                if dy > 0:
                        scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, dy, scale=scale, delta=delta)

                total_weight = x_weight + y_weight
                weight_x = x_weight / total_weight
                weight_y = y_weight / total_weight

                scharr_x_abs = cv2.convertScaleAbs(scharr_x)
                scharr_y_abs = cv2.convertScaleAbs(scharr_y)
                combined = cv2.addWeighted(scharr_x_abs, weight_x, scharr_y_abs, weight_y, 0)
                return combined

        def sharpen_filter(
                self,
                image: Optional[np.ndarray] = None,
                kernel: Optional[np.ndarray] = None,
        ) -> np.ndarray:
                """
                Sharpen the source image using a convolution kernel.

                Args:
                        image (Optional[np.ndarray]): Original image to enhance.
                        kernel (Optional[np.ndarray]): Custom sharpening kernel. If omitted,
                                a 3x3 sharpen kernel is used.

                Returns:
                        np.ndarray: Sharpened image.
                """
                source = self._resolve_image(image)
                if kernel is None:
                        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
                return cv2.filter2D(source, -1, kernel)

        def relief_filter(
                self,
                image: Optional[np.ndarray] = None,
                kernel: Optional[np.ndarray] = None,
        ) -> np.ndarray:
                """
                Create a relief-style image from the source by applying a directional edge kernel.

                Args:
                        image (Optional[np.ndarray]): Original image to transform.
                        kernel (Optional[np.ndarray]): Convolution kernel for the relief effect.
                                If omitted, a default relief kernel is used.

                Returns:
                        np.ndarray: Relief-processed image.
                """
                source = self._resolve_image(image)
                if source.ndim == 3:
                        source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
                if kernel is None:
                        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
                return cv2.filter2D(source, -1, kernel)

        def old_photo_filter(
                self,
                image: Optional[np.ndarray] = None,
                blur_ksize: tuple[int, int] = (5, 5),
        ) -> np.ndarray:
                """
                Simulate an old photo by blending RGB channels with a warm-toned transfer and
                slight Gaussian blurring.

                Args:
                        image (Optional[np.ndarray]): Original image to transform.
                        blur_ksize (tuple[int, int]): Gaussian blur kernel size for the final soft effect.

                Returns:
                        np.ndarray: Vintage-style image.
                """
                source = self._resolve_image(image)
                if source.ndim != 3:
                        raise ValueError("old_photo_filter requires a color image")

                if len(blur_ksize) != 2 or not all(isinstance(size, int) and size > 0 for size in blur_ksize):
                        raise ValueError("blur_ksize must be a tuple of two positive integers")

                b_channel, g_channel, r_channel = cv2.split(source)
                r_out = np.clip(
                        r_channel * 0.393 + g_channel * 0.769 + b_channel * 0.689,
                        0,
                        255,
                ).astype(np.uint8)
                g_out = np.clip(
                        r_channel * 0.349 + g_channel * 0.686 + b_channel * 0.168,
                        0,
                        255,
                ).astype(np.uint8)
                b_out = np.clip(
                        r_channel * 0.272 + g_channel * 0.534 + b_channel * 0.131,
                        0,
                        255,
                ).astype(np.uint8)

                faded = cv2.merge((b_out, g_out, r_out))
                return cv2.GaussianBlur(faded, blur_ksize, 0)

        def sketch_filter(
                self,
                image: Optional[np.ndarray] = None,
                blur_ksize: tuple[int, int] = (19, 19),
                scale: int = 256,
        ) -> np.ndarray:
                """
                Convert the source image into a sketch-like effect by dividing the grayscale image
                by a Gaussian-blurred version of itself.

                Args:
                        image (Optional[np.ndarray]): Original image to process.
                        blur_ksize (tuple[int, int]): Gaussian blur kernel size for the sketch effect.
                        scale (int): Scaling factor for the division operation.

                Returns:
                        np.ndarray: Sketch-style image in grayscale.
                """
                source = self._resolve_image(image)
                gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if source.ndim == 3 else source

                if len(blur_ksize) != 2 or not all(isinstance(size, int) and size > 0 for size in blur_ksize):
                        raise ValueError("blur_ksize must be a tuple of two positive integers")
                if scale <= 0:
                        raise ValueError("scale must be greater than zero")

                blur = cv2.GaussianBlur(gray, blur_ksize, 0)
                sketch = cv2.divide(gray, blur, scale=scale)
                return sketch

        def count_adjacent_pixel_differences(self) -> dict[int, int]:
                """
                Count the frequency of absolute grayscale intensity differences between
                adjacent pixels in the image.

                Returns:
                        dict[int, int]: Dictionary mapping each difference value to its count.

                Example:
                        diff_counts = detector.count_adjacent_pixel_differences()
                """
                flattened_image = self.image.astype(np.int16).ravel()
                pixel_differences = np.abs(np.diff(flattened_image))
                difference_counts = {difference: 0 for difference in range(256)}
                unique_differences, counts = np.unique(pixel_differences, return_counts=True)
                for difference, count in zip(unique_differences, counts):
                        difference_counts[int(difference)] = int(count)
                return difference_counts


if __name__ == "__main__":
        """
        Example usage for the EdgeDetector class.

        This section is only executed when the file is run directly.
        """
        img=cv2.imread("./gallery/image.jpg")
        h,w=img.shape[:2]
        img=cv2.resize(img,(w//2,h//2))
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #old photo effect
        b,g,r=cv2.split(img)
        r=np.clip(r*0.393+g*0.769+b*0.689,0,255).astype(np.uint8)
        g=np.clip(r*0.349+g*0.686+b*0.168,0,255).astype(np.uint8)
        b=np.clip(r*0.272+g*0.534+b*0.131,0,255).astype(np.uint8)
        #scharr edge detection
        img2=cv2.merge((b,g,r))
        sx=cv2.Scharr(gray,cv2.CV_64F,1,0)
        sy=cv2.Scharr(gray,cv2.CV_64F,0,1)
        img3=cv2.addWeighted(cv2.convertScaleAbs(sx),0.5,cv2.convertScaleAbs(sy),0.5,0)
        #sharpen and relief effect
        kernel_sharpen=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        kernel_relief=np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
        img4=cv2.filter2D(img,-1,kernel_sharpen)
        img5=cv2.filter2D(gray,-1,kernel_relief)
        #sketch effect
        blur=cv2.GaussianBlur(gray,(19,19),0)
        sketch=cv2.divide(gray,blur,scale=256)
        
       
        # print(f"Contour count: {len(contours)}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


