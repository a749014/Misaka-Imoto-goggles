import cv2
import numpy as np
from typing import List, Optional, Tuple

class GaussianEffect:
    """
    A class to generate a sequence of images with varying degrees of Gaussian blur.
    
    This is useful for creating transitions (e.g., fade out) or simulating focus changes.
    """
    
    def __init__(self, src: np.ndarray) -> None:
        """
        Initialize the GaussianEffect with a source image.

        Args:
            src (np.ndarray): The source image (BGR or Grayscale).
        
        Example:
            >>> img = cv2.imread("image.jpg")
            >>> ge = GaussianEffect(img)
        """
        if not isinstance(src, np.ndarray):
            raise TypeError(f"src must be of type np.ndarray, got {type(src)}")
        self.src = src
    def lighter(
        self,
        src: Optional[np.ndarray],
        lightening_amount: int,
    ) -> np.ndarray:
        """
        Lighten or darken the input image by a signed pixel offset.

        Args:
            src (Optional[np.ndarray]): The source image to be lightened. If None,
                use the image stored in self.src.
            lightening_amount (int): The value to add to each pixel. Positive values
                lighten and negative values darken; it must be between -255 and 255.

        Returns:
            np.ndarray: The adjusted image with pixel values limited to 0-255.

        Example:
            >>> img = cv2.imread("image.jpg")
            >>> ge = GaussianEffect(img)
            >>> adjusted_img = ge.lighter(img, -50)  # Darken by 50
        """
        if src is not None and not isinstance(src, np.ndarray):
            raise TypeError(f"src must be of type np.ndarray or None, got {type(src)}")
        if not isinstance(lightening_amount, int) or not (-255 <= lightening_amount <= 255):
            raise ValueError(
                "lightening_amount must be an integer between -255 and 255, "
                f"got {lightening_amount}"
            )

        source = self.src if src is None else src
        if lightening_amount >= 0:
            return cv2.add(source, lightening_amount)
        else:
            return cv2.subtract(source, abs(lightening_amount))

    def generate_gaussian_effects(
        self, 
        ksize: Tuple[int, int], 
        return_images: int, 
        effect: str, 
        sigmaX: float = 0, 
        sigmaY: float = 0,
        lighter: int = 0,
        src: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        Generate a list of images with progressively changing Gaussian blur.

        Args:
            ksize (Tuple[int, int]): The base kernel size (width, height) for the blur.
                                    Must be positive odd numbers, e.g., (15, 15).
            return_images (int): The number of images (frames) to generate in the sequence.
            effect (str): The type of blur progression. 
                          Options:
                          - 'progressive_blur': Linear increase of blur intensity.
                          - 'ease_in_out_blur': Non-linear (quadratic) increase, smoother looking.
            sigmaX (float, optional): Gaussian kernel standard deviation in X direction. Default is 0.
            sigmaY (float, optional): Gaussian kernel standard deviation in Y direction. Default is 0.
            lighter (int, optional): Signed brightness adjustment applied progressively
                                    to the blurred images. Positive values lighten the
                                    images, negative values darken them, and 0 disables
                                    brightness adjustment.
            src (Optional[np.ndarray]): Image to process, typically loaded with
                                       cv2.imread. If None, use self.src.

        Returns:
            List[np.ndarray]: A list containing the blurred images.

        Example:
            >>> ge = GaussianEffect(img)
            >>> # Generate 10 frames with progressive blur
            >>> frames = ge.generate_gaussian_effects((31, 31), 10, 'progressive_blur')
        """
        if src is not None and not isinstance(src, np.ndarray):
            raise TypeError(f"src must be of type np.ndarray or None, got {type(src)}")

        source = self.src if src is None else src
        images: List[np.ndarray] = []
        
        # Get image dimensions to constrain kernel size
        h, w = source.shape[:2]
        if return_images>1:
            for i in range(return_images):
                scale_cefct = 1.0 # Default coefficient

                # Determine scaling coefficient based on selected effect
                if effect == 'progressive_blur':
                    # Linear increase: 0 -> almost 1
                    scale_cefct = i / return_images
                elif effect == 'ease_in_out_blur':
                    # Quadratic increase (parabola shape): 0 -> 1 (peaked at end)
                    scale_cefct = -1 * i * i / (return_images * return_images) + 2 * i / return_images
                else:
                    print(f"Warning: Invalid effect '{effect}'. Scale coefficient will turn to default 1.")

                # Calculate raw kernel size
                core_x = int(ksize[0] * scale_cefct)
                core_y = int(ksize[1] * scale_cefct)

                # Constraint 1: Ensure minimum size is 1 (OpenCV requires positive size)
                if core_x < 1: core_x = 1
                if core_y < 1: core_y = 1

                # Constraint 2: Kernel size must be odd numbers for GaussianBlur
                if core_x % 2 == 0: core_x += 1
                if core_y % 2 == 0: core_y += 1
                
                # Constraint 3: Do not exceed source image dimensions
                # kernel width (core_x) <= image width (w)
                # kernel height (core_y) <= image height (h)
                if core_x > w:
                    # Find largest odd number <= w
                    core_x = w if w % 2 != 0 else w - 1
                if core_y > h:
                    core_y = h if h % 2 != 0 else h - 1
                    
                # Safety check again in case image size was small (e.g., w=0, 1, 2)
                if core_x < 1: core_x = 1
                if core_y < 1: core_y = 1

                # Apply blur
                blurred_img = cv2.GaussianBlur(source, (core_x, core_y), sigmaX, sigmaY)
                if lighter != 0:
                    blurred_img=self.lighter(blurred_img, int(lighter * scale_cefct))  # Apply darkening effect if specified
                images.append(blurred_img)
        else:
            # If return_images is 1, just return the original image (or lightened/darkened if specified)
            final_img = cv2.GaussianBlur(source, ksize, sigmaX, sigmaY)
            if lighter != 0:
                final_img = self.lighter(final_img, lighter)
            return [final_img]
            
        return images

# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy image
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[:] = (100, 150, 200)
    cv2.putText(img, "Gaussian Test", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Initialize class
    gaussian_fx = GaussianEffect(img)

    # Generate 20 frames of 'progressive_blur'
    frames_progressive = gaussian_fx.generate_gaussian_effects(
        ksize=(51, 51), 
        return_images=20, 
        effect='progressive_blur',
        lighter=50  # Lighten progressively
    )

    # Generate 20 frames of 'ease_in_out_blur'
    frames_ease = gaussian_fx.generate_gaussian_effects(
        ksize=(51, 51), 
        return_images=20, 
        effect='ease_in_out_blur',
        lighter=-50  # Darken progressively
    )

    print(f"Generated {len(frames_progressive)} progressive frames.")
    print(f"Generated {len(frames_ease)} ease-in-out frames.")
    for i, frame in enumerate(frames_progressive):
        cv2.imshow(f'Progressive Frame ', frame)
        cv2.waitKey(100)  # Display each frame for 100 ms
    cv2.waitKey(0)  # Wait for a key press before showing the next set of frames
    for i, frame in enumerate(frames_ease):
        cv2.imshow(f'Ease-in-out Frame', frame)
        cv2.waitKey(100)  # Display each frame for 100 ms

