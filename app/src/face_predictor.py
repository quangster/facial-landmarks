import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import numpy as np
import cv2


class FacePredictor:
    def __init__(self, model_path: str="../weights/detector.tflite"):
        options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=model_path)
        )
        self.face_predictor = vision.FaceDetector.create_from_options(options)

    def predict(self, image: np.ndarray):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image
        )
        results = self.face_predictor.detect(mp_image)
        bboxs = []
        for detection in results.detections:
            bbox = detection.bounding_box
            bboxs.append([
                bbox.origin_x, bbox.origin_y,
                bbox.width, bbox.height
            ])
        return bboxs
    
if __name__ == "__main__":
    import cv2

    face_predictor = FacePredictor()

    image = cv2.imread("../assets/images/a.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bboxs = face_predictor.predict(image)
    print(bboxs)