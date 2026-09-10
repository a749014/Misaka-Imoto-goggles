"""OpenCV menu descriptions and renderable menu implementations."""

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from gaussianeffect import GaussianEffect


class _Menu:
	"""Store menu geometry and buttons without creating a graphical widget."""

	def __init__(
		self,
		width: float,
		position: tuple[float, float],
		rounded: bool = False,
		corner_radius: float | None = None,
		length: float = 640,
		button_spacing: float = 8,
		orientation: str = "horizontal",
	) -> None:
		"""Create a menu description.

		Parameters:
			width: Menu width. It must be greater than zero. It is the extent
				*across* the button flow: the vertical thickness of a
				horizontal menu, or the horizontal thickness of a vertical
				menu.
			position: Menu placement as an ``(x, y)`` coordinate. For a
				non-screen-attached menu this is its center point.
			rounded: Whether the menu corners are rounded when it is not
				attached to the screen.
			corner_radius: Corner radius, which cannot exceed half the width.
			length: Menu length. It is the extent *along* the button flow:
				horizontal for a horizontal menu, vertical for a vertical
				menu.
			button_spacing: Gap between neighbouring buttons in pixels.
			orientation: ``"horizontal"`` lines buttons up left to right;
				``"vertical"`` stacks them top to bottom.

		Returns:
			None. The constructor initializes the menu properties.

		Example:
			menu = _Menu(240, (320, 80), False, True, 16, 480)
			side_menu = _Menu(140, (1200, 360), length=480, orientation="vertical")
		"""
		if width <= 0:
			raise ValueError("width must be greater than zero")
		if rounded:
			if corner_radius is None:
				corner_radius = 0
			if corner_radius < 0 or corner_radius > width / 2:
				raise ValueError("corner_radius must be between 0 and half width")
		else:
			corner_radius = None
		if length <= 0:
			raise ValueError("length must be greater than zero")
		if button_spacing < 0:
			raise ValueError("button_spacing cannot be negative")
		if orientation not in ("horizontal", "vertical"):
			raise ValueError("orientation must be 'horizontal' or 'vertical'")

		self.width = width
		self.position = position
		self.rounded = rounded
		self.corner_radius = corner_radius
		self.length = length
		self.button_spacing = button_spacing
		self.orientation = orientation

	def render(self, canvas: tuple[int, int] | np.ndarray, buttons: dict[int, dict[str, Any]]) -> np.ndarray:
		"""Render the menu onto a new or caller-provided BGR canvas.

		Parameters:
			canvas: Either a ``(width, height)`` tuple, which creates a black
				canvas, or a BGR ``numpy.ndarray`` supplied by the caller.
			buttons: User-owned records containing ``button_class`` and
				``button_sequence`` fields.

		Returns:
			A BGR OpenCV image containing the rendered menu.

		Example:
			image = menu.render((1280, 720), buttons)
			black_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
			image = menu.render(black_canvas, buttons)
		"""
		if isinstance(canvas, tuple):
			if len(canvas) != 2 or any(size <= 0 for size in canvas):
				raise ValueError("canvas size must contain two positive values")
			canvas_image = np.zeros((canvas[1], canvas[0], 3), dtype=np.uint8)
		elif isinstance(canvas, np.ndarray):
			if canvas.dtype != np.uint8:
				raise ValueError("canvas must be a uint8 BGR image with shape (height, width, 3)")
			if canvas.ndim == 2:
				canvas_image = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
			elif canvas.ndim == 3 and canvas.shape[2] == 1:
				canvas_image = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
			elif canvas.ndim == 3 and canvas.shape[2] == 3:
				canvas_image = canvas
			else:
				raise ValueError("canvas must be a uint8 BGR image with shape (height, width, 3)")
		else:
			raise TypeError("canvas must be a (width, height) tuple or a BGR numpy array")
		return self._render_into(canvas_image, buttons)

	def _render_into(self, canvas: np.ndarray, buttons: dict[int, dict[str, Any]]) -> np.ndarray:
		"""Render this menu into an existing canvas for subclasses to reuse."""
		raise NotImplementedError("use SingleColorMenu or GaussianblurMenu")


class button:
	"""Store the visual properties and selection behavior of one button."""

	def __init__(
		self,
		icon_path: str,
		button_name: str,
		rounded: bool = False,
		corner_radius: float = 0,
		gaussian_blur_ksize: int = 0,
		lightening: float = 0,
		color: tuple[int, int, int] | None = None,
		button_side_length: int = 64,
		text_size: float = 0.45,
		on_select: Callable[[], Any] | None = None,
	) -> None:
		"""Create a button description and an optional selection callback.

		Parameters:
			icon_path: Path to the button icon image.
			button_name: Textual name of the button.
			rounded: Whether the button has rounded corners.
			corner_radius: Radius of rounded corners; it must not be negative.
			gaussian_blur_ksize: Gaussian blur kernel size; zero disables it.
			lightening: Degree to which the button image is made lighter.
			color: Optional solid BGR color. When set, image effects are skipped.
			button_side_length: Button side length in pixels; it must be positive.
			text_size: OpenCV font scale used for the button name; it must be positive.
			on_select: Function called by ``select`` after selection. It can
				also be assigned later by user code.

		Returns:
			None. The constructor initializes the button properties.

		Example:
			item = button("open.png", "Open", on_select=open_file)
		"""
		if rounded and corner_radius < 0:
			raise ValueError("corner_radius cannot be negative")
		if not rounded:
			corner_radius = 0
		if gaussian_blur_ksize < 0 or gaussian_blur_ksize % 2 == 0 and gaussian_blur_ksize != 0:
			raise ValueError("gaussian_blur_ksize must be zero or a positive odd integer")
		if button_side_length <= 0:
			raise ValueError("button_side_length must be greater than zero")
		if text_size <= 0:
			raise ValueError("text_size must be greater than zero")

		self.icon_path = icon_path
		self.button_name = button_name
		self.rounded = rounded
		self.corner_radius = corner_radius
		self.gaussian_blur_ksize = gaussian_blur_ksize
		self.lightening = lightening
		self.color = color
		self.button_side_length = button_side_length
		self.text_size = text_size
		self.on_select = on_select

	def select(self) -> Any:
		"""Execute the user-defined callback after this button is selected.

		Parameters:
			None. The callback is read from the ``on_select`` property.

		Returns:
			The callback's return value, or ``None`` when no callback exists.

		Example:
			item.on_select = open_file
			result = item.select()
		"""
		if self.on_select is None:
			return None
		return self.on_select()


class SingleColorMenu(_Menu):
	"""Render a menu with a single solid BGR background color.

	All geometry options of ``_Menu``, including ``orientation``, are accepted.
	"""

	def __init__(self, *args: Any, color: tuple[int, int, int], **kwargs: Any) -> None:
		"""Create a solid-color menu and initialize inherited menu properties.

		Parameters:
			color: Menu background color in OpenCV BGR order.
			args and kwargs: Arguments accepted by ``_Menu``.

		Returns:
			None. The menu properties are initialized.

		Example:
			menu = SingleColorMenu(80, (640, 50), True, color=(40, 90, 160))
			side_menu = SingleColorMenu(90, (1200, 360), color=(40, 90, 160), orientation="vertical")
		"""
		super().__init__(*args, **kwargs)
		self.color = _validate_color(color)

	def _render_into(self, canvas: np.ndarray, buttons: dict[int, dict[str, Any]]) -> np.ndarray:
		"""Draw the solid menu and its buttons onto ``canvas``."""
		return _render_menu(self, canvas, buttons, self.color, None, 0)


class GaussianblurMenu(_Menu):
	"""Render a menu whose image buttons receive Gaussian blur and lighting.

	All geometry options of ``_Menu``, including ``orientation``, are accepted.
	"""

	def __init__(
		self,
		*args: Any,
		ksize: int = 5,
		lightening: int = 0,
		**kwargs: Any,
	) -> None:
		"""Create a Gaussian-effect menu.

		Parameters:
			ksize: Odd positive Gaussian kernel size.
			lightening: Signed brightness adjustment from -255 to 255.
			args and kwargs: Arguments accepted by ``_Menu``.

		Returns:
			None. The menu properties are initialized.

		Example:
			menu = GaussianblurMenu(90, (640, 50), True, ksize=9)
			side_menu = GaussianblurMenu(90, (1200, 360), ksize=9, orientation="vertical")
		"""
		super().__init__(*args, **kwargs)
		if ksize <= 0 or ksize % 2 == 0:
			raise ValueError("ksize must be a positive odd integer")
		if not -255 <= lightening <= 255:
			raise ValueError("lightening must be between -255 and 255")
		self.ksize = ksize
		self.lightening = lightening

	def _render_into(self, canvas: np.ndarray, buttons: dict[int, dict[str, Any]]) -> np.ndarray:
		"""Draw the menu background; callers render buttons separately."""
		return _render_menu(self, canvas, buttons, (70, 70, 70), self.ksize, self.lightening)


@lru_cache(maxsize=256)
def _load_icon(path: str) -> np.ndarray:
	"""Load and cache one icon image from disk."""
	image = cv2.imread(path, cv2.IMREAD_COLOR)
	if image is None:
		raise FileNotFoundError(f"button image not found: {path}")
	image.flags.writeable = False
	return image


@lru_cache(maxsize=256)
def _scaled_icon(path: str, width: int, height: int) -> np.ndarray:
	"""Resize one icon once and cache the output for reuse."""
	icon = _load_icon(path)
	resized = cv2.resize(icon, (width, height), interpolation=cv2.INTER_AREA)
	resized.flags.writeable = False
	return resized


@lru_cache(maxsize=256)
def _rounded_mask(height: int, width: int, radius: int) -> np.ndarray:
	"""Build a reusable rounded-rectangle mask for menu and button corners."""
	mask = np.zeros((height, width), dtype=np.uint8)
	if radius <= 0:
		mask[:] = 255
		return mask
	cv2.rectangle(mask, (radius, 0), (width - radius - 1, height - 1), 255, -1)
	cv2.rectangle(mask, (0, radius), (width - 1, height - radius - 1), 255, -1)
	for center, start_angle, end_angle in (
		((radius, radius), 180, 270),
		((width - radius - 1, radius), 270, 360),
		((radius, height - radius - 1), 90, 180),
		((width - radius - 1, height - radius - 1), 0, 90),
	):
		cv2.ellipse(mask, center, (radius, radius), 0, start_angle, end_angle, 255, -1)
	mask.flags.writeable = False
	return mask


@lru_cache(maxsize=256)
def _cached_text_size(text: str, font_scale: float, thickness: int = 1) -> tuple[tuple[int, int], int]:
	"""Cache font metrics so repeated button labels do not recalculate layout."""
	return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)


def _validate_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
	"""Validate and return a three-channel BGR color tuple."""
	if len(color) != 3 or any(not isinstance(channel, int) or not 0 <= channel <= 255 for channel in color):
		raise ValueError("color must contain three integer channels from 0 to 255")
	return color


def _render_menu(
	menu: _Menu,
	canvas: np.ndarray,
	buttons: dict[int, dict[str, Any]],
	menu_color: tuple[int, int, int],
	ksize: int | None,
	lightening: int,
) -> np.ndarray:
	"""Render only the menu background onto a BGR canvas.

	Parameters:
		menu: Menu object containing geometry, orientation, corner settings,
			spacing, and buttons.
		canvas: Destination BGR image. Its shape is used as the available page size.
		buttons: Retained for API compatibility; render buttons separately.
		menu_color: Background color in OpenCV's BGR channel order.
		ksize: Menu-level Gaussian kernel size. ``None`` disables image effects,
			which is used by ``SingleColorMenu``.
		lightening: Signed brightness adjustment passed to ``GaussianEffect``.

	Returns:
		The same canvas object after only the menu background has been rendered.
		Call ``render_buttons`` separately to add buttons.

	Raises:
		ValueError: If the menu is too small to contain its buttons with the
			configured icon, label, and spacing in the chosen orientation.

	Example:
		canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
		result = _render_menu(menu, canvas, (70, 70, 70), 9, 20)
	"""
	canvas_height, canvas_width = canvas.shape[:2]
	# The menu length is explicit, and position is always its center coordinate.
	# Orientation decides which canvas axis carries the length: a horizontal
	# menu runs length-wise along x, a vertical menu runs length-wise along y.
	# The width is always the extent across the button flow.
	menu_length = int(menu.length)
	menu_width = int(menu.width)
	vertical = menu.orientation == "vertical"
	if vertical:
		x = int(menu.position[0] - menu_width / 2)
		y = int(menu.position[1] - menu_length / 2)
		menu_span_x = menu_width
		menu_span_y = menu_length
	else:
		x = int(menu.position[0] - menu_length / 2)
		y = int(menu.position[1] - menu_width / 2)
		menu_span_x = menu_length
		menu_span_y = menu_width
	# Solid menus keep their configured color. Gaussian menus sample the covered
	# canvas area because their effect is defined relative to the live image.
	if ksize is None:
		menu_image = np.full((menu_span_y, menu_span_x, 3), menu_color, dtype=np.uint8)
	else:
		menu_image = canvas[max(y, 0):min(y + menu_span_y, canvas_height), max(x, 0):min(x + menu_span_x, canvas_width)].copy()
	canvas_x_start = max(x, 0)
	canvas_y_start = max(y, 0)
	canvas_x_end = min(x + menu_span_x, canvas_width)
	canvas_y_end = min(y + menu_span_y, canvas_height)
	if canvas_x_start >= canvas_x_end or canvas_y_start >= canvas_y_end:
		return canvas
	if ksize is not None:
		menu_image = cv2.GaussianBlur(menu_image, (ksize, ksize), 0)
		if lightening:
			menu_image = cv2.convertScaleAbs(menu_image, alpha=1.0, beta=float(lightening))
	if menu.rounded:
		# Reuse a cached mask rather than rebuilding the same rounded geometry on every frame.
		radius = int(menu.corner_radius)
		mask = _rounded_mask(menu_span_y, menu_span_x, radius)
		canvas_region = canvas[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end]
		src_y0 = max(0, -y)
		src_x0 = max(0, -x)
		src_y1 = src_y0 + canvas_region.shape[0]
		src_x1 = src_x0 + canvas_region.shape[1]
		mask_region = mask[src_y0:src_y1, src_x0:src_x1]
		menu_region = menu_image[src_y0:src_y1, src_x0:src_x1]
		canvas_region[mask_region > 0] = menu_region[mask_region > 0]
	else:
		canvas_region = canvas[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end]
		src_y0 = max(0, -y)
		src_x0 = max(0, -x)
		src_y1 = src_y0 + canvas_region.shape[0]
		src_x1 = src_x0 + canvas_region.shape[1]
		visible = menu_image[src_y0:src_y1, src_x0:src_x1]
		if visible.size == 0 or canvas_region.size == 0:
			return canvas
		canvas_region[:] = visible

	# Buttons retain their configured dimensions. The largest button extent
	# along the layout axis is used as the step so later buttons cannot
	# overlap earlier ones, and each button is centered across the layout
	# axis exactly like the horizontal menu centered buttons vertically.
	return canvas


def render_buttons(
	menu: _Menu,
	canvas: np.ndarray,
	buttons: dict[int, dict[str, Any]],
	ksize: int | None = None,
	lightening: int = 0,
) -> list[dict[str, Any]]:
	"""Render ordered buttons and return records that were placed.

	Parameters:
		menu: Menu geometry used to position the buttons.
		canvas: Menu-sized BGR image to modify in place.
		buttons: User-owned records containing ``button_class`` and
			``button_sequence`` fields.
		ksize: Optional Gaussian kernel used for image buttons.
		lightening: Signed brightness adjustment for image buttons.

	Returns:
		A list of successfully rendered records in sequence order. Each record
		contains ``button_class``, ``button_sequence``, and ``button_position``.
		The canvas is modified in place. Buttons that do not fit are skipped.

	Example:
		rendered_buttons = render_buttons(menu, menu_image, buttons)
	"""
	# ---- Module 1: initialize geometry and layout info ----
	canvas_height, canvas_width = canvas.shape[:2]
	menu_width = int(menu.width)
	menu_length = int(menu.length)
	vertical = menu.orientation == "vertical"
	#print(
	# 	f"[render_buttons][1-initialize] canvas={canvas_width}x{canvas_height}, "
	# 	f"menu_width={menu_width}, menu_length={menu_length}, "
	# 	f"orientation={'vertical' if vertical else 'horizontal'}, "
	# 	f"button_spacing={menu.button_spacing}"
	# )

	# ---- Module 2: sort buttons ----
	ordered_records = sorted(buttons.items(), key=lambda pair: pair[1]["button_sequence"])
	#print(
	# 	f"[render_buttons][2-sort] total {len(ordered_records)} buttons, "
	# 	f"order: {[record['button_class'].button_name for _, record in ordered_records]}"
	# )

	# ---- Module 3: compute button sizes and spacing ----
	rendered_buttons: list[dict[str, Any]] = []
	max_button_width = max(
		(_button_render_width(record[1]["button_class"]) for record in ordered_records),
		default=0,
	)
	max_button_height = max(
		(_button_render_height(record[1]["button_class"]) for record in ordered_records),
		default=0,
	)
	button_step = max_button_height if vertical else max_button_width
	#print(
	# 	f"[render_buttons][3-size] max_button_width={max_button_width}, "
	# 	f"max_button_height={max_button_height}, button_step={button_step}"
	# )

	# ---- Module 4: layout and render each button ----
	for index, (key, record) in enumerate(ordered_records):
		item = record["button_class"]
		button_width = _button_render_width(item)
		button_height = _button_render_height(item)
		button_x = (menu_width - button_width) // 2 if vertical else int(
			menu.button_spacing + index * (button_step + menu.button_spacing)
		)
		button_y = int(menu.button_spacing + index * (button_step + menu.button_spacing)) if vertical else (menu_width - button_height) // 2
		#print(
		# 	f"[render_buttons][4-layout] #{index} '{item.button_name}': "
		# 	f"size={button_width}x{button_height}, pos=({button_x}, {button_y})"
		# )

		# ---- Module 4a: menu boundary checks ----
		if vertical:
			if button_width > menu_width or button_y + button_height > menu_length or button_x < 0 or button_y < 0:
				#print(
				# 	f"[render_buttons][4a-boundary] Button {item.button_name} does not fit "
				# 	f"in vertical menu (w={button_width}/{menu_width}, "
				# 	f"y+h={button_y + button_height}/{menu_length}); skipping"
				# )
				break
		else:
			if button_height > menu_width or button_x + button_width > menu_length or button_x < 0 or button_y < 0:
				#print(
				# 	f"[render_buttons][4a-boundary] Button {item.button_name} does not fit "
				# 	f"in horizontal menu (h={button_height}/{menu_width}, "
				# 	f"x+w={button_x + button_width}/{menu_length}); skipping"
				# )
				break

		# ---- Module 4b: canvas boundary checks ----
		if button_y + button_height > canvas_height or button_x + button_width > canvas_width:
			print(
				f"[render_buttons][4b-canvas] Button {item.button_name} exceeds canvas "
				f"(needs {button_x + button_width}x{button_y + button_height}, "
				f"canvas {canvas_width}x{canvas_height}); stopping"
			)
			break

		# ---- Module 4c: create button image and write it to the canvas ----
		background_crop = canvas[button_y:button_y + button_height, button_x:button_x + button_width]
		button_image = _button_image(item, ksize, lightening, background_crop)
		canvas[button_y:button_y + button_image.shape[0], button_x:button_x + button_image.shape[1]] = button_image
		rendered_buttons.append({
			"button_class": item,
			"button_sequence": record["button_sequence"],
			"button_position": (button_x, button_y),
		})
		#print(
		# 	f"[render_buttons][4c-render] '{item.button_name}' drawn, "
		# 	f"image size={button_image.shape[1]}x{button_image.shape[0]}"
		# )

	# ---- Module 5: summary ----
	#print(
	# 	f"[render_buttons][5-summary] rendered {len(rendered_buttons)}/{len(ordered_records)} buttons: "
	# 	f"{[(r['button_class'].button_name, r['button_position']) for r in rendered_buttons]}"
	# )
	return rendered_buttons



def mark_button(
	canvas: np.ndarray,
	button_position: tuple[int, int],
	position: str = "top",
	color: tuple[int, int, int] = (255, 255, 255),
	radius: int = 5,
) -> np.ndarray:
	"""Draw a colored selection marker beside a button.

	Parameters:
		canvas: BGR image to modify in place.
		button_position: Button top-left coordinate in ``canvas``.
		position: Marker side: ``"top"``, ``"bottom"``, ``"left"``, or ``"right"``.
		color: Marker color in BGR order.
		radius: Marker radius in pixels; it must be positive.

	Returns:
		The same canvas after drawing the marker.

	Example:
		mark_button(image, (40, 80), "top", (255, 255, 255))
	"""
	if position not in ("top", "bottom", "left", "right"):
		raise ValueError("position must be 'top', 'bottom', 'left', or 'right'")
	if radius <= 0:
		raise ValueError("radius must be positive")
	_validate_color(color)
	x, y = button_position
	marker_x = max(radius, min(canvas.shape[1] - radius - 1, x + 20))
	marker_y = max(radius, min(canvas.shape[0] - radius - 1, y + 20))
	if position == "top":
		point = (marker_x, max(radius, y - radius - 2))
	elif position == "bottom":
		point = (marker_x, min(canvas.shape[0] - radius - 1, y + radius + 2))
	elif position == "left":
		point = (max(radius, x - radius - 2), marker_y)
	else:
		point = (min(canvas.shape[1] - radius - 1, x + radius + 2), marker_y)
	cv2.circle(canvas, point, radius, color, -1, cv2.LINE_AA)
	return canvas


def _button_render_width(item: button) -> int:
	"""Return the rendered button width based on its shape and side length.

	Parameters:
		item: Button whose geometry is being measured.

	Returns:
		The image width in pixels, reduced for a rounded button as configured.

	Example:
		width = _button_render_width(button("icon.png", "Open"))
	"""
	return max(1, item.button_side_length - int(item.corner_radius)) if item.rounded else item.button_side_length


def _button_render_height(item: button) -> int:
	"""Return the total rendered button height, including its text label.

	Parameters:
		item: Button whose geometry is being measured.

	Returns:
		The image height plus the reserved label area in pixels.

	Example:
		height = _button_render_height(button("icon.png", "Open"))
	"""
	(_, text_height), baseline = _cached_text_size(item.button_name, item.text_size, 1)
	return _button_render_width(item) + text_height + baseline + 6


def _button_image(
	item: button,
	ksize: int | None,
	lightening: int,
	background_image: np.ndarray,
) -> np.ndarray:
	"""Create one complete button image, including its icon and label.

	Parameters:
		item: Button containing image, shape, color, and label settings.
		ksize: Menu-level Gaussian kernel, or ``None`` for no menu effect.
		lightening: Menu-level signed brightness adjustment.
		background_image: Already processed menu-background pixels behind this
			button. These pixels fill the label area and rounded corners.

	Returns:
		A BGR OpenCV image whose top area contains the icon or solid color and
		whose bottom padding contains the button name at ``item.text_size``.

		When ``item.color`` is configured, no file is read and neither Gaussian
		blur nor brightness adjustment is applied. This gives explicit button
		colors precedence over the Gaussian menu effect.

	Example:
		background = np.full((88, 52, 3), 70, dtype=np.uint8)
		image = _button_image(button("icon.png", "Open"), 5, 10, background)
	"""
	button_width = _button_render_width(item)
	button_height = button_width
	if item.color is not None:
		# A color-configured button is intentionally independent of its icon path
		# and bypasses all image processing as required by the button contract.
		button_background = np.full(
			(button_height, button_width, 3),
			_validate_color(item.color),
			dtype=np.uint8,
		)
		icon = button_background.copy()
	else:
		# Read and resize the source image to the configured icon area. The
		# lru-cached resize avoids re-reading and rescaling the same icon every frame.
		icon = _scaled_icon(str(Path(item.icon_path)), button_width, button_height).copy()
		button_background = icon
	if item.rounded:
		# Apply the rounded shape only to the icon area. The label remains a
		# rectangular text strip underneath the icon.
		radius = min(int(item.corner_radius), button_width // 2, button_height // 2)
		mask = _rounded_mask(button_height, button_width, radius)
		# Pixels outside the mask must expose the menu background. Keeping the
		# source image there would make the corners look square.
		rounded_icon = background_image[:button_height, :button_width].copy()
		rounded_icon[mask > 0] = icon[mask > 0]
		icon = rounded_icon
	# Derive the label strip from the actual font metrics. This keeps the
	# button height correct when callers choose a larger or smaller text size.
	(text_width, text_height), baseline = _cached_text_size(item.button_name, item.text_size, 1)
	label_height = text_height + baseline + 6
	# Initialize the complete button with the selected color or processed image.
	# This keeps the label strip and rounded corners from becoming black.
	result = np.empty((button_height + label_height, button_width, 3), dtype=np.uint8)
	result[:] = background_image[:result.shape[0], :result.shape[1]]
	result[:button_height] = icon
	if item.color is not None:
		# A colored button only colors its icon area. Its label remains over the
		# processed menu background, just like the outside of a rounded icon.
		result[button_height:] = background_image[button_height:result.shape[0], :button_width]
	result[:button_height] = icon
	text_x = max(0, (button_width - text_width) // 2)
	# Center text when it fits; max(0, ...) keeps long labels inside the image
	# coordinate system instead of producing a negative drawing origin.
	cv2.putText(result, item.button_name, (text_x, button_height + text_height + 3), cv2.FONT_HERSHEY_SIMPLEX, item.text_size, (255, 255, 255), 1, cv2.LINE_AA)
	return result

