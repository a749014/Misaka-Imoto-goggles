"""Small example showing how to create and render a multi-button menu."""

import cv2

from menu_bar import GaussianblurMenu, button, mark_button, render_buttons
import numpy as np


def open_action() -> None:
	"""Print the action associated with the Open button.

	Returns:
		None. The function only demonstrates a user-defined callback.

	Example:
		open_action()
	"""
	print("Open selected")


def close_action() -> None:
	"""Print the action associated with the Close button.

	Returns:
		None. The function only demonstrates a user-defined callback.

	Example:
		close_action()
	"""
	print("Close selected")


def main() -> None:
	"""Create a Gaussian menu with six buttons and save its rendered image.

	Returns:
		None. The rendered BGR image is written to ``menu_example.png``.

	Example:
		main()
	"""
	menu = GaussianblurMenu(
		width=96,
		position=(240, 320),
		length=600,
		rounded=True,
		corner_radius=20,
		button_spacing=16,
		ksize=101,
		lightening=0,
		orientation="vertical",
	)
	buttons = {
		"open": {
			"button_class": button(
			"gallery/image.jpg",
			"Open",
			rounded=True,
			corner_radius=12,
			button_side_length=64,
			text_size=0.45,
			on_select=open_action,
		),
			"button_sequence": 1,
		},
		"close": {
			"button_class": button(
			None,
			"Close",
			rounded=True,
			corner_radius=12,
			button_side_length=64,
			text_size=0.45,
			color=(80, 120, 210),
			on_select=close_action,
		),
			"button_sequence": 2,
		},
		"gallery": {
			"button_class": button(
				None,
				"Gallery",
				rounded=True,
				corner_radius=12,
				button_side_length=64,
				text_size=0.45,
				color=(60, 150, 80),
				on_select=open_action,
			),
			"button_sequence": 3,
		},
		"detect": {
			"button_class": button(
				None,
				"Detect",
				rounded=True,
				corner_radius=12,
				button_side_length=64,
				text_size=0.45,
				color=(180, 100, 50),
				on_select=open_action,
			),
			"button_sequence": 4,
		},
		"settings": {
			"button_class": button(
				None,
				"Settings",
				rounded=True,
				corner_radius=12,
				button_side_length=64,
				text_size=0.45,
				color=(120, 80, 180),
				on_select=close_action,
			),
			"button_sequence": 5,
		},
		"exit": {
			"button_class": button(
				None,
				"Exit",
				rounded=True,
				corner_radius=12,
				button_side_length=64,
				text_size=0.45,
				color=(60, 70, 190),
				on_select=close_action,
			),
			"button_sequence": 6,
		},
	}
	black_canvas = cv2.imread("gallery/image2.png")
	black_canvas = cv2.resize(black_canvas, (480, 640))
	selected_index = 0
	confirmed_index: int | None = None
	while True:
		image = menu.render(black_canvas.copy(), buttons)
		menu_origin = (
			int(menu.position[0] - menu.width / 2),
			int(menu.position[1] - menu.length / 2),
		)
		menu_region = image[
			menu_origin[1]:menu_origin[1] + int(menu.length),
			menu_origin[0]:menu_origin[0] + int(menu.width),
		]
		rendered_buttons = render_buttons(menu, menu_region, buttons, menu.ksize, menu.lightening)
		print(rendered_buttons)
		if rendered_buttons:
			selected_index = min(selected_index, len(rendered_buttons) - 1)
			for index, rendered_button in enumerate(rendered_buttons):
				button_position = rendered_button["button_position"]
				canvas_position = (
					button_position[0] + menu_origin[0],
					button_position[1] + menu_origin[1],
				)
				if index == confirmed_index:
					mark_button(image, canvas_position, "bottom", (0, 0, 255))
				elif index == selected_index:
					mark_button(image, canvas_position, "top", (255, 255, 255))
		cv2.imshow("Menu Example", image)
		key = cv2.waitKey(0) & 0xFF
		if key in (27, ord("q")):
			break
		if key in (ord("w"), 2490368):
			selected_index = max(0, selected_index - 1)
		elif key in (ord("s"), 2621440):
			selected_index = min(len(rendered_buttons) - 1, selected_index + 1)
		elif key in (13, 10) and rendered_buttons:
			confirmed_index = selected_index
			rendered_buttons[confirmed_index]["button_class"].select()
	# print("Rendered menu to menu_example.png")
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()