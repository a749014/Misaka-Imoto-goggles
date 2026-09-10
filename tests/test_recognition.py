import numpy as np

from recognition import Model, format_model_menu_label, get_model_names


def test_format_model_menu_label_uses_index_and_total_for_canvas_text():
    label = format_model_menu_label("Face Detection Model (Default)", 0, 17)

    assert label == "(1/17) Face Detection Model (Default)"


def test_default_model_is_available_without_loading_classifier():
    model = Model("DEFAULT")

    assert model.model_name == "DEFAULT"
    assert model.classifier is None
    assert model.error_message == ""
    assert model.recognize(np.zeros((20, 20, 3), dtype=np.uint8), display=False) == []


def test_missing_model_reports_error_without_exception():
    model = Model("THIS_MODEL_DOES_NOT_EXIST")

    assert model.model_name == "THIS_MODEL_DOES_NOT_EXIST"
    assert model.classifier is None
    assert "not found" in model.error_message.lower() or "does not exist" in model.error_message.lower()


def test_default_is_in_model_list():
    names = get_model_names()

    assert "DEFAULT" in names
    assert names[0] == "DEFAULT"
