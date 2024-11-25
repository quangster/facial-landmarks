import onnx
import onnxruntime
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions


class FaceKeypointsPredictor:
    def __init__(self, model_path: str="../weights/mobilenetv3.onnx"):
        # Checking
        temp = onnx.load(model_path)
        onnx.checker.check_model(temp)
        
        # Load model
        self.predictor = onnxruntime.InferenceSession(model_path)
    
    def predict(self, image: np.ndarray):
        width, height, _ = image.shape
        image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        image = image / 255.0
        image = image.astype(np.float32)
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        
        ort_input = {self.predictor.get_inputs()[0].name: image}
        ort_out = self.predictor.run(None, ort_input)
        landmarks = ort_out[0][0]
        landmarks += 0.5
        landmarks *= np.array([height, width])
        return landmarks
    
    def detect_landmarks(self, image: np.ndarray, bboxs: list[list[int]]):
        

        # def process_bbox(bbox):
        #     start_point = bbox[0], bbox[1]
        #     end_point = bbox[0] + bbox[2], bbox[1] + bbox[3]
        #     crop_face = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
        #     landmarks = self.predict(crop_face)
        #     landmarks += np.array([start_point[0], start_point[1]])
        #     landmarks = landmarks.astype(int)
        #     return landmarks

        # with concurrent.futures.ThreadPoolExecutor(max) as executor:
        #     all_landmarks = list(executor.map(process_bbox, bboxs))

        all_landmarks = []
        for bbox in bboxs:
            start_point = bbox[0], bbox[1]
            end_point = bbox[0] + bbox[2], bbox[1] + bbox[3]
            crop_face = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
            landmarks = self.predict(crop_face)
            landmarks += np.array([start_point[0], start_point[1]])
            landmarks = landmarks.astype(int)
            all_landmarks.append(landmarks)

        return all_landmarks
    
    def annotate_face_keypoints(
        self, 
        image: np.ndarray, 
        bboxs: list[list[int]]=None,
        show_bbox: bool=True
    ):
        for bbox in bboxs:
            start_point = bbox[0], bbox[1]
            end_point = bbox[0] + bbox[2], bbox[1] + bbox[3]
            crop_face = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
            landmarks = self.predict(crop_face)
            landmarks += np.array([start_point[0], start_point[1]])
            landmarks = landmarks.astype(int)
            image.flags.writeable = True
            if show_bbox:
                cv2.rectangle(image, start_point, end_point, (0, 255, 0), 3)
            for i, p in enumerate(landmarks):
                # cv2.putText(image, str(i), (p[0], p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.circle(image, p, 2, (0, 255, 0), -1)
        return image

class FaceMesh:
    def __init__(self, model_path: str="../weights/face_landmarker.task"):
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=5,
        )
        self.predictor = vision.FaceLandmarker.create_from_options(options)

    def predict(self, image: np.ndarray):
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image.flags.writeable = False
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        return self.predictor.detect(mp_image)
    
    def annotate_face_keypoints(
        self, 
        image: np.ndarray, 
        bboxs: list[list[int]],
        show_bbox: bool=True
    ):
        image.flags.writeable = False
        results = self.predict(image)
        image.flags.writeable = True
        # Draw bounding box
        if show_bbox:
            for bbox in bboxs:
                start_point = bbox[0], bbox[1]
                end_point = bbox[0] + bbox[2], bbox[1] + bbox[3]
                cv2.rectangle(image, start_point, end_point, (0, 255, 0), 3)
        # Draw face landmarks
        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
                ])

                solutions.drawing_utils.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks_proto,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_tesselation_style())
                solutions.drawing_utils.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks_proto,
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_contours_style())
                solutions.drawing_utils.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks_proto,
                    connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_iris_connections_style())
        return image


if __name__ == "__main__":
    from face_predictor import FacePredictor

    # Init models
    face_predictor = FacePredictor()
    # keypoints_predictor = FaceKeypointsPredictor("../weights/mobilenetv3.onnx")
    keypoints_predictor = FaceMesh()

    # Image
    image = cv2.imread("../assets/images/a.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Predict bounding box
    bboxs = face_predictor.predict(image)
    print(bboxs)

    annotated_image = keypoints_predictor.annotate_face_keypoints(image, bboxs, show_bbox=True)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 800, 600)
    cv2.imshow("Image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
