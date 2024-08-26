from transformers import AutoTokenizer
import torch
torch.manual_seed(1234)

#from ipex_llm.transformers import AutoModelForCausalLM
from pathlib import Path
import time
import gradio as gr

import random
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from ipex_llm import optimize_model
import numpy as np

model_path = "Qwen/Qwen-VL-Chat"
# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# use xpu device
# Load model
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", trust_remote_code=True)
model = optimize_model(model, 
                       low_bit='sym_int4', 
                       modules_to_not_convert=['c_fc', 'out_proj'])
model = model.to('xpu')
# Specify hyperparameters for generation (No need to do this if you are using transformers>=4.32.0)
model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)

vl_chat_history = None
query = None
initial = True


with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Qwen-VL-chat demo on ARC iGPU
    """)
    with gr.Row():
        with gr.Column():
            imagebox = gr.Image(type="filepath")
            
        with gr.Column():
    	    chatbot = gr.Chatbot(height=800)
    
    with gr.Row():
        with gr.Column():
            gr.HTML("&nbsp;")                    
        with gr.Column():
            msg = gr.Textbox(label='Prompt')
                
    with gr.Row():
        with gr.Column():
            clear = gr.Button("Clear")
        with gr.Column():
            submit = gr.Button("Submit")

    def user(text, history, files):
        global query
        print(f"um: {text}, file: {files}, history: {history}")
        history = [] if history is None else history
        history.append([(files,),None])
        history.append([text,None])
        print(f"n_history: {history}")
        
        query = tokenizer.from_list_format([
	    {'image': files},
	    #{'text': '这是什么'},
	    {'text': text},	
	])
        
        return "", history

    def bot(history):
        print(f"history: {history}")
        global vl_chat_history, query
        response, vl_chat_history = model.chat(tokenizer, query=query, history=vl_chat_history)
        print(response)       
        #bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
        #time.sleep(2)
        history[-1][1] = response
 
        image = tokenizer.draw_bbox_on_latest_picture(response, vl_chat_history)
        if image:
           #print(f"image: {image.get_image()}")
           image.save('temp.jpg')
           temp_img = Path('temp.jpg')
           history.append([None, (temp_img.as_posix(),)])

        #history = (bot_message, None)
        return history
        
    def clearfn():
        global vl_chat_history, query
        vl_chat_history = None
        query = None
        return []

    chat_msg = msg.submit(user, [msg, chatbot, imagebox], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot, api_name="bot_response"
    )
    
    chat_msg = submit.click(fn=user, inputs=[msg, chatbot, imagebox], outputs=[msg, chatbot]).then(
        bot, chatbot, chatbot, api_name="bot_response"
    )
    
    #chat_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [msg])
    clear.click(fn=clearfn, outputs=chatbot)

    examples = gr.Examples(
       examples=[
         ["./examples/demo.jpeg", '这是什么'],
         ["./examples/demo.jpeg", '输出"击掌"的检测框'],
       ],
       inputs=[imagebox, msg],
    )

#demo.launch(server_name="0.0.0.0", server_port=443, ssl_verify=False, ssl_certfile="c:/test/cert.pem", ssl_keyfile="c:/test/key.pem")
demo.launch(server_name="0.0.0.0", server_port=50000)


# 1st dialogue turn
#query = tokenizer.from_list_format([
#    {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
#    #{'text': '这是什么'},
#    {'text': 'describe the image'},
#
#])
#response, history = model.chat(tokenizer, query=query, history=None)
#print(response)
# 图中是一名年轻女子在沙滩上和她的狗玩耍，狗的品种可能是拉布拉多。她们坐在沙滩上，狗的前腿抬起来，似乎在和人类击掌。两人之间充满了信任和爱。

# 2nd dialogue turn
#response, history = model.chat(tokenizer, '输出"击掌"的检测框', history=history)
#print(response)
# <ref>??</ref><box>(517,508),(589,611)</box>
#image = tokenizer.draw_bbox_on_latest_picture(response, history)
#if image:
#  image.save('1.jpg')
#else:
#  print("no box")