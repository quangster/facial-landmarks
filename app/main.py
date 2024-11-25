import gradio as gr
from gradio_webrtc import WebRTC
import numpy as np
import cv2
from src import FacePredictor, FaceKeypointsPredictor, FaceMesh, FaceFilter

def process_image(
    input_image: np.ndarray,
    model_choice: str,
    option: str,
    show_bbox: bool,
    filter_choice: str,
):
    # Face landmarks
    if option == "Landmarks":
        if model_choice == "facemesh":
            return annotate_landmark_using_facemesh(input_image, show_bbox)
        else:
            return annotate_landmark_using_mymodel(input_image, model_choice, show_bbox)
    # Filter
    elif option == "Filter":
        return apply_filter(input_image, model_choice, filter_choice)
    else:
        if show_bbox:
            global face_predictor
            input_image.flags.writeable = False
            bboxs = face_predictor.predict(input_image)
            input_image.flags.writeable = True
            for bbox in bboxs:
                start_point = bbox[0], bbox[1]
                end_point = bbox[0] + bbox[2], bbox[1] + bbox[3]
                cv2.rectangle(input_image, start_point, end_point, (0, 255, 0), 3)
        return input_image

def annotate_landmark_using_facemesh(image: np.ndarray, show_bbox: bool):
    global facemesh
    global face_predictor

    image.flags.writeable = False
    bboxs = face_predictor.predict(image)
    return facemesh.annotate_face_keypoints(image, bboxs, show_bbox)

def annotate_landmark_using_mymodel(image: np.ndarray, model_type: str="mobilenetv3", show_bbox: bool=True):
    global face_predictor
    global mobilenet_keypoints_predictor
    global resnet_keypoints_predictor

    image.flags.writeable = False
    # choose model
    if model_type == "mobilenetv3":
        model = mobilenet_keypoints_predictor
    else:
        model = resnet_keypoints_predictor
    # get bounding boxs
    bboxs = face_predictor.predict(image)
    # using bboxs to crop face and annotate face landmarks
    return model.annotate_face_keypoints(image, bboxs, show_bbox)

def apply_filter(image: np.ndarray, model_type: str="mobilenetv3", filter_choice: str="anonymous"):
    global face_predictor
    global mobilenet_keypoints_predictor
    global resnet_keypoints_predictor
    global facefilter

    image.flags.writeable = False

    # choose model
    if model_type == "mobilenetv3":
        model = mobilenet_keypoints_predictor
    else:
        model = resnet_keypoints_predictor
    # get bounding boxs
    bboxs = face_predictor.predict(image)
    all_landmarks = model.detect_landmarks(image, bboxs)
    return facefilter.apply_filter(image, bboxs, all_landmarks, filter_choice)

def process_stream(
    input_image: np.ndarray,
    model_choice: str,
    option_cam: str,
    show_bbox: bool,
    filter_choice: str,
):
    image = process_image(input_image, model_choice, option_cam, show_bbox, filter_choice)
    return cv2.flip(image, 1)

def create_interface():
    models = ["mobilenetv3", "resnet18", "facemesh"]
    filters = ["anonymous", "jason-joker", "dog", "cat"]
    image_examples = [
        "assets/images/a.jpg",
        "assets/images/b.jpg",
        "assets/images/c.jpeg",
    ]

    with gr.Blocks(title="Filter App") as interface:
        with gr.Tab("Image"):
            # UI
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Input")
                    model_choice = gr.Dropdown(label="Select Model", choices=models, value=models[0])
                    option = gr.Radio(label="Choose Option", choices=["Landmarks", "Filter", "None"], value="Landmarks")
                    show_bbox = gr.Checkbox(label="Show bounding box", value=True, visible=True)
                    filter_choice = gr.Dropdown(label="Select Filter", choices=filters, value=filters[0], visible=False)
                    submit_button = gr.Button(value="Submit", variant="primary")
                    clear_button = gr.ClearButton(value="Clear")
                with gr.Column():
                    output_image = gr.Image(label="Output")
                    gr.Examples(examples=image_examples, inputs=[input_image])

            # Logic
            def update_options(choice):
                if choice == "Filter":
                    return gr.update(visible=False), gr.update(visible=True)
                else:
                    return gr.update(visible=True), gr.update(visible=False)
                
            option.change(update_options, inputs=[option], outputs=[show_bbox, filter_choice])
            submit_button.click(fn=process_image, inputs=[input_image, model_choice, option, show_bbox, filter_choice], outputs=[output_image])
            clear_button.click(fn=lambda: (None, None), inputs=[], outputs=[input_image, output_image])


        with gr.Tab("Camera"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_choice_cam = gr.Dropdown(label="Select Model", choices=models, value=models[0])
                    option_cam = gr.Radio(label="Choose Option", choices=["Landmarks", "Filter", "None"], value="Landmarks")
                    show_bbox_cam = gr.Checkbox(label="Show bounding box", value=True, visible=True)
                    filter_choice_cam = gr.Dropdown(label="Select Filter", choices=filters, value=filters[0], visible=False)
                    option_cam.change(update_options, inputs=[option_cam], outputs=[show_bbox_cam, filter_choice_cam])
                with gr.Column(scale=3):
                    video = WebRTC(label="Stream", width=700, height=700)
                    video.stream(
                        fn=process_stream, 
                        inputs=[video, model_choice_cam, option_cam, show_bbox_cam, filter_choice_cam], 
                        outputs=[video],
                    )

        gr.Markdown("Created by [quangster](https://github.com/quangster)")
        
    return interface

if __name__ == "__main__":
    # Init models
    face_predictor = FacePredictor("./weights/detector.tflite")
    mobilenet_keypoints_predictor = FaceKeypointsPredictor("./weights/mobilenetv3.onnx")
    resnet_keypoints_predictor = FaceKeypointsPredictor("./weights/resnet18.onnx")
    facemesh = FaceMesh("./weights/face_landmarker.task")
    facefilter = FaceFilter("./assets/filters")

    # Launch app
    interface = create_interface()
    interface.launch(server_port=7860)
