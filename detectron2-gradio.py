from pathlib import Path
import time
import gradio as gr

import random
from PIL import Image
import detectron2.model_zoo as detectron_zoo
from detectron2.modeling import GeneralizedRCNN
from detectron2.export import TracingAdapter
import detectron2.data.transforms as T
from detectron2.data import detection_utils
from detectron2.structures import Instances, Boxes
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

import torch
import openvino as ov
import warnings
from typing import List, Dict

import numpy as np
import requests

def get_model_and_config(model_name: str):
    """
    Helper function for downloading PyTorch model and its configuration from Detectron2 Model Zoo

    Parameters:
      model_name (str): model_id from Detectron2 Model Zoo
    Returns:
      model (torch.nn.Module): Pretrained model instance
      cfg (Config): Configuration for model
    """
    cfg = detectron_zoo.get_config(model_name + ".yaml", trained=True)
    model = detectron_zoo.get(model_name + ".yaml", trained=True)
    return model, cfg

def convert_detectron2_model(model: torch.nn.Module, sample_input: List[Dict[str, torch.Tensor]]):
    """
    Function for converting Detectron2 models, creates TracingAdapter for making model tracing-friendly,
    prepares inputs and converts model to OpenVINO Model

    Parameters:
      model (torch.nn.Module): Model object for conversion
      sample_input (List[Dict[str, torch.Tensor]]): sample input for tracing
    Returns:
      ov_model (ov.Model): OpenVINO Model
    """
    # prepare input for tracing adapter
    tracing_input = [{"image": sample_input[0]["image"]}]

    # override model forward and disable postprocessing if required
    if isinstance(model, GeneralizedRCNN):

        def inference(model, inputs):
            # use do_postprocess=False so it returns ROI mask
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

    else:
        inference = None  # assume that we just call the model directly

    # create traceable model
    traceable_model = TracingAdapter(model, tracing_input, inference)
    warnings.filterwarnings("ignore")
    # convert PyTorch model to OpenVINO model
    ov_model = ov.convert_model(traceable_model, example_input=sample_input[0]["image"])
    return ov_model
 
def get_sample_inputs(image_path, cfg):
     # get a sample data
     original_image = detection_utils.read_image(image_path, format=cfg.INPUT.FORMAT)
     # Do same preprocessing as DefaultPredictor
     aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
     height, width = original_image.shape[:2]
     image = aug.get_transform(original_image).apply_image(original_image)
     image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
 
     inputs = {"image": image, "height": height, "width": width}
 
     # Sample ready
     sample_inputs = [inputs]
     return sample_inputs
    
def postprocess_detection_result(outputs: Dict, orig_height: int, orig_width: int, conf_threshold: float = 0.0):
    """
    Helper function for postprocessing prediction results

    Parameters:
      outputs (Dict): OpenVINO model output dictionary
      orig_height (int): original image height before preprocessing
      orig_width (int): original image width before preprocessing
      conf_threshold (float, optional, defaults 0.0): confidence threshold for valid prediction
    Returns:
      prediction_result (instances): postprocessed predicted instances
    """
    boxes = outputs[0]
    classes = outputs[1]
    has_mask = len(outputs) >= 5
    masks = None if not has_mask else outputs[2]
    scores = outputs[2 if not has_mask else 3]
    model_input_size = (
        int(outputs[3 if not has_mask else 4][0]),
        int(outputs[3 if not has_mask else 4][1]),
    )
    filtered_detections = scores >= conf_threshold
    boxes = Boxes(boxes[filtered_detections])
    scores = scores[filtered_detections]
    classes = classes[filtered_detections]
    out_dict = {"pred_boxes": boxes, "scores": scores, "pred_classes": classes}
    if masks is not None:
        masks = masks[filtered_detections]
        out_dict["pred_masks"] = torch.from_numpy(masks)
    instances = Instances(model_input_size, **out_dict)
    return detector_postprocess(instances, orig_height, orig_width)


def draw_instance_prediction(img: np.ndarray, results: Instances, cfg: "Config"):
    """
    Helper function for visualization prediction results

    Parameters:
      img (np.ndarray): original image for drawing predictions
      results (instances): model predictions
      cfg (Config): model configuration
    Returns:
       img_with_res: image with results
    """
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    visualizer = Visualizer(img, metadata, instance_mode=ColorMode.IMAGE)
    img_with_res = visualizer.draw_instance_predictions(results)
    return img_with_res
    
    
MODEL_DIR = Path("model")
DATA_DIR = Path("examples")
TEMP_DIR = Path("temp")

MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

input_image_url = "https://farm9.staticflickr.com/8040/8017130856_1b46b5f5fc_z.jpg"

image_file = DATA_DIR / "example_image.jpg"

if not image_file.exists():
    image = Image.open(requests.get(input_image_url, stream=True).raw)
    image.save(image_file)

   
core = ov.Core()
devices = []
print(f"available inference device: {core.available_devices}")

devices.extend(core.available_devices)
device="CPU"  #"NPU"
task = 0 # 0=detection, 1=segmentation
conf_threshold = 0.8

    
def run_detection(image_file, device):
    global MODEL_DIR, TEMP_DIR, core, conf_threshold
    model_name = "COCO-Detection/faster_rcnn_R_50_FPN_1x"
    model, cfg = get_model_and_config(model_name)
    sample_input = get_sample_inputs(image_file, cfg)

    model_xml_path = MODEL_DIR / (model_name.split("/")[-1] + ".xml")
    if not model_xml_path.exists():
        ov_model = convert_detectron2_model(model, sample_input)
        ov.save_model(ov_model, MODEL_DIR / (model_name.split("/")[-1] + ".xml"))
    else:
        ov_model = model_xml_path

    compiled_model = core.compile_model(ov_model, device)
    image=Image.open(image_file)
    results = compiled_model(sample_input[0]["image"])
    results = postprocess_detection_result(results, sample_input[0]["height"], sample_input[0]["width"], conf_threshold=conf_threshold)
    img_with_res = draw_instance_prediction(np.array(image), results, cfg)
    outimg=Image.fromarray(img_with_res.get_image())
    outimg.save(TEMP_DIR / "temp.jpg")
    return TEMP_DIR / "temp.jpg"
    
def run_segmentation(image_file, device):
    global MODEL_DIR, TEMP_DIR, core, conf_threshold
    model_name = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x"
    model, cfg = get_model_and_config(model_name)
    sample_input = get_sample_inputs(image_file, cfg)
    
    model_xml_path = MODEL_DIR / (model_name.split("/")[-1] + ".xml")
    
    if not model_xml_path.exists():
        ov_model = convert_detectron2_model(model, sample_input)
        ov.save_model(ov_model, MODEL_DIR / (model_name.split("/")[-1] + ".xml"))
    else:
        ov_model = model_xml_path
    
    compiled_model = core.compile_model(ov_model, device)    
    image=Image.open(image_file)    
    results = compiled_model(sample_input[0]["image"])
    results = postprocess_detection_result(results, sample_input[0]["height"], sample_input[0]["width"], conf_threshold=conf_threshold)
    img_with_res = draw_instance_prediction(np.array(image), results, cfg)
    outimg=Image.fromarray(img_with_res.get_image())
    outimg.save(TEMP_DIR / "temp.jpg")
    return TEMP_DIR / "temp.jpg"    

def set_task(task_selector):
    global task
    task_name = ["Detection", "Segmentation"]
    
    if task_selector == "Detection":
        task=0
    elif task_selector == "Segmentation":
        task=1
    else:
        task=0 #default
        
    print(f"Selected task: {task}")
    
    return gr.Dropdown(info=f"selected task: {task_name[task]}")

def set_confidence(conf_slider):
    global conf_threshold
    conf_threshold = conf_slider
    
def set_accel(accel_selector):
    global device
    
    device=accel_selector
    print(f"Selected accel: {device}")
    
    return gr.Dropdown(info=f"selected accel: {device}")
    
with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Detectron2 demo on ARC iGPU
    """)
    with gr.Row():
        with gr.Column():
            imagebox = gr.Image(type="filepath")
            with gr.Row():
                with gr.Column():
                    task_select=gr.Dropdown(
                         choices=["Detection", "Segmentation"], label="Select Task", value="Detection")
            with gr.Row():
                with gr.Column():
                    accel_select=gr.Dropdown(
                         choices=devices, label="Select Accelerator", value="CPU")
                         
            with gr.Row():
                with gr.Column():                         
                    with gr.Accordion("Parameters", open=False, visible=True) as parameter_row:
                        confidence = gr.Slider(
                                             minimum=0.0,
                                             maximum=1.0,
                                             value=0.8,
                                             step=0.05,
                                             interactive=True,
                                             label="Confidence",
                                             info="Choose value between 0-1."
                                             )
                             
        with gr.Column():
    	    chatbot = gr.Chatbot(height=800)
               
    with gr.Row():
        with gr.Column():
            clear_btn = gr.Button("Clear")
        with gr.Column():
            submit_btn = gr.Button("Submit")

    def user(history, files):
        global query
        print(f"file: {files}, history: {history}")
        history = [] if history is None else history
        history.append([(files,),None])
        history.append(["detect and segment the above image",None])
        print(f"n_history: {history}")
        
        #query = tokenizer.from_list_format([
	#    {'image': files},
	#    #{'text': '这是什么'},
	#    {'text': text},	
	#])
        
        return history

    def bot(history, files):
        global device, task
        print(f"history: {history}")
        #global vl_chat_history, query
        #response, vl_chat_history = model.chat(tokenizer, query=query, history=vl_chat_history)
        #print(response)
        if task == 0:
            response = run_detection(files, device)
        else:
            response = run_segmentation(files, device)
        history[-1][1] = (response,)
 
        #image = tokenizer.draw_bbox_on_latest_picture(response, vl_chat_history)
        #if image:
        #   #print(f"image: {image.get_image()}")
        #   image.save('temp.jpg')
        #   temp_img = Path('temp.jpg')
        #   history.append([None, (temp_img.as_posix(),)])

        #history = (bot_message, None)
        return history
        
    def clearfn():
        global query
        query = None
        return []
    
    chat_msg = submit_btn.click(fn=user, inputs=[chatbot, imagebox], outputs=[chatbot]).then(
        bot, inputs=[chatbot, imagebox], outputs=chatbot, api_name="bot_response"
    )
    
    #chat_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [msg])
    clear_btn.click(fn=clearfn, outputs=chatbot)
    
    task_select.change(set_task, inputs=task_select, outputs=task_select)
    accel_select.change(set_accel, inputs=accel_select, outputs=accel_select)
 
    confidence.change(set_confidence, inputs=confidence)

    examples = gr.Examples(
       examples=[
         ["./examples/demo.jpeg"],
         ["./examples/demo.jpeg"],
       ],
       inputs=[imagebox],
    )

#demo.launch(server_name="0.0.0.0", server_port=443, ssl_verify=False, ssl_certfile="c:/test/cert.pem", ssl_keyfile="c:/test/key.pem")
demo.launch(server_name="0.0.0.0", server_port=50000)

