#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import requests
from PIL import Image
from transformers import (
    LlavaNextForConditionalGeneration,
    AutoTokenizer,
    LlavaNextProcessor,
    #CLIPImageProcessor,
    #TextStreamer,
    TextIteratorStreamer,
    #AutoProcessor,
)
#from transformers.feature_extraction_utils import BatchFeature
#from intel_npu_acceleration_library.compiler import CompilerConfig
#import intel_npu_acceleration_library
import torch
from ipex_llm import optimize_model
import numpy as np

import gradio as gr
from threading import Event, Thread
import os

cur_dir = None
#checkpoint = "Intel/llava-gemma-2b"
#checkpoint = "llava-hf/llava-1.5-7b-hf"
checkpoint = "llava-hf/llama3-llava-next-8b-hf"

title_markdown = """
# 🌋 LLaVA: Large Language and Vision Assistant
"""

tos_markdown = """
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
"""


# Load model
model = LlavaNextForConditionalGeneration.from_pretrained(checkpoint)
model = optimize_model(model, 
                       low_bit='sym_int4')
model = model.half().to('xpu')

processor = LlavaNextProcessor.from_pretrained(checkpoint)


def change_model(selectbox):
    global checkpoint
    print(f"select change: {selectbox}")
    checkpoint = selectbox
    return gr.Dropdown(info=f"[model loading ...]"),gr.Textbox(interactive=False),gr.Button(interactive = False)

def apply_new_model():
    global checkpoint
    global model
    global processor
    model = LlavaNextForConditionalGeneration.from_pretrained(checkpoint)
    model = optimize_model(model, 
                           low_bit='sym_int4')
    model = model.half().to('xpu')
    processor = LlavaNextProcessor.from_pretrained(checkpoint)    

    return gr.Dropdown(info=f"[model loaded successfully]"),gr.Textbox(interactive=True),gr.Button(interactive = True)

def clear_history(textbox, imagebox, chatbot):
    """
    callback function for clearing chat windows in interface on clear button click

    Params:
      textbox: current textbox for user messages state
      imagebox: current imagebox state
      chatbot: current chatbot state
    Returns:
      empty textbox, imagebox and chatbot states
    """
    #conv.messages = []

    return None, None, None
    
def user(message, history):
    """
    callback function for updating user messages in interface on submit button click

    Params:
      message: current message
      history: conversation history
    Returns:
      updated message and conversation history
    """
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]
    

def bot(image, history, temperature=0.2, top_p=0.7, max_new_tokens=1024):
    """
    callback function for running chatbot on submit button click

    Params:
      history: conversation history
      temperature:  parameter for control the level of creativity in AI-generated text.
                    By adjusting the `temperature`, you can influence the AI model's probability distribution, making the text more focused or diverse.
      top_p: parameter for control the range of tokens considered by the AI model based on their cumulative probability.

    """

    text = history[-1][0]
    print(f"prompt: {text}")

    prompt = processor.apply_chat_template(
        [{
            "role": "user", 
            "content": [
              {"type": "text", "text": f"{text}"},
              {"type": "image"},
            ],
         },],
        add_generation_prompt=True,
        tokenize=False
    )
    with torch.inference_mode():
      inputs = processor(images=image, text=prompt, return_tensors="pt").to('xpu')
      streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
      
    #generate_kwargs = dict(
    #    input_ids=input_ids,
    #    pixel_values=image_tensor,
    #    max_new_tokens=max_new_tokens,
    #    temperature=temperature,
    #    do_sample=temperature > 0.001,
    #    top_p=top_p,
    #    streamer=streamer,
    #    use_cache=True,
    #    #stopping_criteria=[stopping_criteria],
    #)
    
    stream_complete = Event()

    def generate_and_signal_complete():
        """
        genration function for single thread
        """
        with torch.inference_mode():
           #output=model.generate(**inputs, max_new_tokens=150, streamer=streamer)
           output=model.generate(**inputs, max_new_tokens=150, streamer=streamer)
           #history[-1][1]=processor.decode(output[0], skip_special_tokens=True)
           stream_complete.set()

    #with torch.inference_mode():
    #       #output=model.generate(**inputs, max_new_tokens=150, streamer=streamer)
    #   output=model.generate(**inputs, max_new_tokens=100)
    #   history[-1][1]=processor.decode(output[0], skip_special_tokens=True)
    #   
    #   print(f"output: {history[-1][1]}")
    #return history


    t1 = Thread(target=generate_and_signal_complete)
    t1.start()

    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        if not new_text:
            continue
        partial_text += new_text
        history[-1][1] = partial_text
        yield history


with gr.Blocks(title="LLaVA-NeXT HF") as demo:
    gr.Markdown(title_markdown)

    with gr.Row():
        with gr.Column():
            imagebox = gr.Image(type="pil")
            with gr.Accordion("Parameters", open=False, visible=True) as parameter_row:
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.2,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    interactive=True,
                    label="Top P",
                )
                max_output_tokens = gr.Slider(
                    minimum=0,
                    maximum=1024,
                    value=512,
                    step=64,
                    interactive=True,
                    label="Max output tokens",
                )
                
            model_select=gr.Dropdown(
                 ["llava-hf/llama3-llava-next-8b-hf",
                  "llava-hf/llava-v1.6-mistral-7b-hf",
                  "llava-hf/llava-v1.6-vicuna-7b-hf",
                  "Efficient-Large-Model/VILA-7b",
                 ], label="Select Model", value=0)
                
        with gr.Column(scale=3):
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(height=400)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox = gr.Textbox(
                            show_label=False,
                            placeholder="Enter text and press ENTER",
                            visible=True,
                            container=False,
                        )
                    with gr.Column(scale=1, min_width=60):
                        submit_btn = gr.Button(value="Submit", visible=True)
                with gr.Row(visible=True) as button_row:
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)
    with gr.Row():
       with gr.Column():
           if cur_dir is None:
              cur_dir = os.path.dirname(os.path.abspath(__file__))
              gr.Examples(examples=[
                [f"{cur_dir}/examples/extreme_ironing.jpg", "What is unusual about this image?"],
                [f"{cur_dir}/examples/waterview.jpg", "What are the things I should be cautious about when I visit here?"],
              ], inputs=[imagebox, textbox])

    gr.Markdown(tos_markdown)

    submit_event = textbox.submit(
        fn=user,
        inputs=[textbox, chatbot],
        outputs=[textbox, chatbot],
        queue=False,
    ).then(
        bot,
        [imagebox, chatbot, temperature, top_p, max_output_tokens],
        chatbot,
        queue=True,
    )
    # Register listeners
    clear_btn.click(clear_history, [textbox, imagebox, chatbot], [chatbot, textbox, imagebox])
    submit_click_event = submit_btn.click(
        fn=user,
        inputs=[textbox, chatbot],
        outputs=[textbox, chatbot],
        queue=False,
    ).then(
        bot,
        [imagebox, chatbot, temperature, top_p, max_output_tokens],
        chatbot,
        queue=True,
    )
    
    model_select.change(change_model, inputs=[model_select], outputs=[model_select, textbox, submit_btn]).then(
                        apply_new_model, outputs=[model_select, textbox, submit_btn])

# if you are launching remotely, specify server_name and server_port
# demo.launch(server_name='your server name', server_port='server port in int')
# Read more in the docs: https://gradio.app/docs/
try:
    demo.queue(max_size=2).launch(debug=True)
except Exception:
    demo.queue(max_size=2).launch(share=True, debug=True)
    
    
