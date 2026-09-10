DEFAULT_MODEL_NAME = "DEFAULT"

MODEL_INFO = [
    {
        "ModelName": "Eye Detection Model",
        "ModelPath": "modules_xml/haarcascade_eye.xml",
        "description": "Detect eye regions using Haar cascade classifiers",
    },
    {
        "ModelName": "Eyeglass Eye Detection Model",
        "ModelPath": "modules_xml/haarcascade_eye_tree_eyeglasses.xml",
        "description": "Detect eyes with glasses using Haar cascade classifiers",
    },
    {
        "ModelName": "Cat Face Detection Model",
        "ModelPath": "modules_xml/haarcascade_frontalcatface.xml",
        "description": "Detect frontal cat faces using Haar cascade classifiers",
    },
    {
        "ModelName": "Enhanced Cat Face Detection Model",
        "ModelPath": "modules_xml/haarcascade_frontalcatface_extended.xml",
        "description": "Detect frontal cat faces using an enhanced Haar cascade",
    },
    {
        "ModelName": "Face Detection Model (Focus on Side Face)",
        "ModelPath": "modules_xml/haarcascade_frontalface_alt.xml",
        "description": "Detect faces using Haar cascade classifiers, suitable for frontal and some side poses",
    },
    {
        "ModelName": "Face Detection Model (Version 2)",
        "ModelPath": "modules_xml/haarcascade_frontalface_alt2.xml",
        "description": "Detect faces using another set of Haar parameters",
    },
    {
        "ModelName": "Face Detection Model (Tree Structure)",
        "ModelPath": "modules_xml/haarcascade_frontalface_alt_tree.xml",
        "description": "Detect frontal faces using a tree-structured Haar cascade",
    },
    {
        "ModelName": "Face Detection Model (Default)",
        "ModelPath": "modules_xml/haarcascade_frontalface_default.xml",
        "description": "Detect frontal faces using OpenCV's default Haar cascade",
    },
    {
        "ModelName": "Full Body Detection Model",
        "ModelPath": "modules_xml/haarcascade_fullbody.xml",
        "description": "Detect full human bodies using Haar cascade classifiers",
    },
    {
        "ModelName": "Left Eye Detection Model",
        "ModelPath": "modules_xml/haarcascade_lefteye_2splits.xml",
        "description": "Detect left eyes using a dual-branch Haar cascade",
    },
    {
        "ModelName": "Russian License Plate Detection Model",
        "ModelPath": "modules_xml/haarcascade_licence_plate_rus_16stages.xml",
        "description": "Detect Russian license plates using a 16-stage Haar cascade",
    },
    {
        "ModelName": "Lower Body Detection Model",
        "ModelPath": "modules_xml/haarcascade_lowerbody.xml",
        "description": "Detect a person's lower body using Haar cascade classifiers",
    },
    {
        "ModelName": "Profile Face Detection Model",
        "ModelPath": "modules_xml/haarcascade_profileface.xml",
        "description": "Detect profile faces using Haar cascade classifiers",
    },
    {
        "ModelName": "Right Eye Detection Model",
        "ModelPath": "modules_xml/haarcascade_righteye_2splits.xml",
        "description": "Detect right eyes using a dual-branch Haar cascade",
    },
    {
        "ModelName": "Russian License Plate Detection Model (Backup)",
        "ModelPath": "modules_xml/haarcascade_russian_plate_number.xml",
        "description": "Detect Russian license plate numbers using Haar cascade classifiers",
    },
    {
        "ModelName": "Smile Detection Model",
        "ModelPath": "modules_xml/haarcascade_smile.xml",
        "description": "Detect smiling faces or smile regions using Haar cascade classifiers",
    },
    {
        "ModelName": "Upper Body Detection Model",
        "ModelPath": "modules_xml/haarcascade_upperbody.xml",
        "description": "Detect a person's upper body using Haar cascade classifiers",
    },
    {
        "ModelName": "bird",
        "ModelPath": "modules_xml/bird.pt",
        "description": "",
    },
]

import gc
import os
import time

import cv2


PATH = os.path.dirname(os.path.abspath(__file__))
def _restore_opencv_threads():
    """ultralytics imports set cv2.setNumThreads(0), which disables all
    OpenCV internal parallelism (incl. detectMultiScale). Restore it."""
    threads = max(1, (os.cpu_count() or 4) - 1)
    if cv2.getNumThreads() != threads:
        cv2.setNumThreads(threads)

def get_model_names():
    """Return the selectable recognition model names with DEFAULT first."""
    names = [DEFAULT_MODEL_NAME]
    names.extend(model_info["ModelName"] for model_info in MODEL_INFO)
    return names


def format_model_menu_label(model_name: str, index: int, total: int) -> str:
    """Build the recognition menu text used on the canvas.

    Example:
        format_model_menu_label("Face Detection Model (Default)", 0, 17)
        -> "(1/17) Face Detection Model (Default)"
    """
    safe_total = max(1, total)
    safe_index = max(0, min(index, safe_total - 1))
    return f"({safe_index + 1}/{safe_total}) {model_name}"


class Model:
    """Wrap Haar cascade model loading and recognition.

    This class resolves a model by name or XML/PT path, loads the classifier,
    and detects targets in a frame.
    """

    _last_yolo_model = None

    @classmethod
    def _release_previous_yolo_model(cls):
        """Release the most recently loaded YOLO model so it does not linger across model switches."""
        previous_model = cls._last_yolo_model
        cls._last_yolo_model = None
        if previous_model is None:
            print("No previous YOLO model to release")
            return

        try:
            print(f"Releasing previous YOLO model: {type(previous_model)}")
            del previous_model
            print("Previous YOLO model released")
        except Exception:
            print("Failed to delete previous YOLO model")
            pass

        # for attr_name in ("model", "predictor", "overrides"):
        #     try:
        #         if hasattr(previous_model, attr_name):
        #             setattr(previous_model, attr_name, None)
        #     except Exception:
        #         pass

        gc.collect()

    def __init__(self, model_name_or_path=DEFAULT_MODEL_NAME):
        """Load a Haar XML classifier or an Ultralytics YOLO model.

        Args:
            model_name_or_path: A MODEL_INFO name, the special ``DEFAULT`` value,
                or an XML/PT model path. ``DEFAULT`` leaves recognition disabled
                without raising an error.

        Returns:
            None. The loaded model is stored on the instance.

        Example:
            model = Model("Smile Detection Model")
        """
        self.model_name = str(model_name_or_path)
        self.model_path = None
        self.is_yolo_model = False
        self.classifier = None
        self.error_message = ""

        if self.model_name.strip().upper() == DEFAULT_MODEL_NAME:
            self.model_name = DEFAULT_MODEL_NAME
            self.model_path = DEFAULT_MODEL_NAME
            self.classifier = None
            self.is_yolo_model = False
            self.error_message = ""
            self._release_previous_yolo_model()
            return

        try:
            self.model_path = self._resolve_model_path(model_name_or_path)
            self.is_yolo_model = self.model_path.lower().endswith(".pt")
            self.classifier = self._create_classifier(self.model_path)
            if self.is_yolo_model:
                if Model._last_yolo_model is not None and Model._last_yolo_model is not self.classifier:
                    self._release_previous_yolo_model()
                Model._last_yolo_model = self.classifier
            else:
                self._release_previous_yolo_model()
        except Exception as exc:  # pragma: no cover - defensive failure path
            self.model_path = None
            self.error_message = f"Model loading failed: {exc}"
            self.classifier = None
            self.is_yolo_model = False

    @staticmethod
    def _resolve_model_path(model_name_or_path):
        """Resolve a model name to the backing XML/PT file path."""
        if model_name_or_path is None:
            raise FileNotFoundError("Model name is empty")

        model_name = str(model_name_or_path).strip()
        if model_name.upper() == DEFAULT_MODEL_NAME:
            raise FileNotFoundError("DEFAULT mode does not load any recognition model")

        model_path = next(
            (
                model_info["ModelPath"]
                for model_info in MODEL_INFO
                if model_info["ModelName"] == model_name
            ),
            model_name,
        )

        if not os.path.isabs(model_path):
            model_path = os.path.join(PATH, model_path)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        return model_path

    @staticmethod
    def _create_classifier(model_path):
        """Create and load a model from an XML or PT path.

        Args:
            model_path: Absolute path to a Haar XML or Ultralytics PT model.

        Returns:
            A loaded ``cv2.CascadeClassifier`` or ``ultralytics.YOLO`` object.

        Example:
            classifier = Model._create_classifier(model_path)
        """
        if model_path.lower().endswith(".pt"):
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise RuntimeError(
                    "Loading a PT model requires ultralytics: pip install ultralytics"
                ) from error
            c= YOLO(model_path)
            _restore_opencv_threads()
            return c

        major_version = int(cv2.__version__.split(".")[0])
        if major_version >= 5:
            # OpenCV 5 separates classifier creation from XML loading; prefer the newer factory APIs.
            objdetect = getattr(cv2, "objdetect", None)
            classifier_factory = getattr(cv2, "CascadeClassifier_create", None)
            if classifier_factory is None and objdetect is not None:
                classifier_factory = getattr(
                    objdetect, "CascadeClassifier_create", None
                )
            if classifier_factory is None:
                classifier_factory = getattr(objdetect, "CascadeClassifier", None)
            if classifier_factory is None:
                classifier_factory = getattr(cv2, "CascadeClassifier", None)
        else:
            classifier_factory = getattr(cv2, "CascadeClassifier", None)

        if classifier_factory is None:
            raise RuntimeError(
                f"Current OpenCV {cv2.__version__} has no available CascadeClassifier interface"
            )

        if major_version >= 5:
            # Newer versions create an empty classifier first and load the XML afterward.
            classifier = classifier_factory()
            if not classifier.load(model_path):
                raise RuntimeError(f"Could not load model file: {model_path}")
        else:
            classifier = classifier_factory(model_path)
        if hasattr(classifier, "empty") and classifier.empty():
            raise RuntimeError(f"Could not load model file: {model_path}")
        return classifier

    def recognize(
        self,
        frame,
        display=True,
        draw_type="rectangle",
        color=(0, 255, 0),
        thickness=1,
        minNeighbors=4,
        scaleFactor=1.1,
    ):
        """Detect objects and optionally draw results on the supplied frame.

        Args:
            frame: The image to detect and, when enabled, annotate in place.
            display: Whether to draw results on ``frame``; no extra window is opened.
            draw_type: Haar shape, either ``rectangle`` or ``circle``. YOLO uses boxes.
            color: BGR drawing color for Haar and YOLO annotations.
            thickness: Line thickness for annotations.
            minNeighbors: Haar detection sensitivity setting.
            scaleFactor: Haar image scale factor, which must be greater than 1.

        Returns:
            Haar detection tuples, or the Ultralytics prediction result list for PT models.

        Example:
            results = model.recognize(frame, color=(0, 0, 255))
        """
        if self.classifier is None and self.model_name == DEFAULT_MODEL_NAME:
            if display:
                self._draw_status_text(frame, "DEFAULT mode: no recognition model loaded", color=(0, 0, 255))
            return []

        if self.classifier is None:
            if display:
                self._draw_status_text(frame, self.error_message or "Model not loaded", color=(0, 0, 255))
            return []

        if draw_type not in ("rectangle", "circle"):
            raise ValueError("draw_type must be 'rectangle' or 'circle'")
        if scaleFactor <= 1:
            raise ValueError("scaleFactor must be greater than 1")

        if self.is_yolo_model:
            # ``show=False`` keeps Ultralytics from opening a second display window.
            detect_results = self.classifier.predict(
                source=frame,
                show=False,
                verbose=False,
            )
            if display:
                self._draw_yolo_results(frame, detect_results, color, thickness)
            return detect_results
        else:
            detect_results = self.classifier.detectMultiScale(
                frame, scaleFactor=scaleFactor, minNeighbors=minNeighbors
            )
        if not display:
            return detect_results

        for x, y, width, height in detect_results:
            if draw_type == "rectangle":
                cv2.rectangle(
                    frame, (x, y), (x + width, y + height), color, thickness
                )
            else:
                center = (x + width // 2, y + height // 2)
                radius = max(width, height) // 2
                cv2.circle(frame, center, radius, color, thickness)
        return detect_results

    @staticmethod
    def _draw_status_text(frame, text, color=(0, 255, 0)):
        """Render a short diagnostic line on the frame without opening extra windows."""
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_yolo_results(self, frame, predict_results, color, thickness):
        """Draw YOLO boxes, class names, and confidence scores on one frame.

        Args:
            frame: The BGR image to annotate in place.
            predict_results: The list returned by Ultralytics ``predict``.
            color: BGR color used for boxes and labels.
            thickness: Box line thickness.

        Returns:
            None. The supplied frame is modified directly.

        Example:
            model._draw_yolo_results(frame, model.classifier.predict(frame), (0, 255, 0), 1)
        """
        if not predict_results:
            return

        result = predict_results[0]
        if result.boxes is None:
            return

        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        names = result.names

        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            x1, y1, x2, y2 = map(int, box)
            class_name = names[class_id]
            label = f"{class_name} {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            text_y = max(y1 - 8, 15)
            cv2.putText(
                frame,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                max(thickness, 1),
                cv2.LINE_AA,
            )


def main():
    """Open the camera and display real-time recognition results."""
    model = Model("Face Detection Model (Version 2)")
    capture = cv2.VideoCapture(0)
    last_t = time.time()

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                continue

            model.recognize(frame, draw_type="rectangle")
            current_t = time.time()
            fps = 1 / (current_t - last_t) if current_t != last_t else 0
            cv2.putText(
                frame,
                f"FPS:{round(fps, 1)}",
                (0, len(frame) // 15),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                1,
            )
            cv2.imshow("camera", frame)
            last_t = current_t

            if cv2.waitKey(1) == 27:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

