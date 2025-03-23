import gradio as gr
from utils.upload_file import UploadFile
from utils.chatbot import ChatAgent

# Create Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 SQL Chatbot - Chat First, Query After Upload")

    chatbot = gr.Chatbot()
    
    with gr.Row():
        input_txt = gr.Textbox(placeholder="Ask a question...", label="Your Message")
        send_button = gr.Button("Send")
        print("Here2")
    print("here1")
    
    upload_btn = gr.UploadButton(
                    "📁 Upload CSV or XLSX files", file_types=['.csv'], file_count="multiple")
                
    print("app upload_btn: ",upload_btn)
    file_msg = upload_btn.upload(fn=UploadFile.run_pipeline, inputs=[
                upload_btn, chatbot], outputs=[input_txt, chatbot], queue=False)

    '''
    txt_msg = input_txt.submit(fn=ChatAgent.respond,
                                       inputs=[chatbot, input_txt],
                                       outputs=[input_txt,
                                                chatbot],
                                       queue=False).then(lambda: gr.Textbox(interactive=True),
                                                         None, [input_txt], queue=False)
    '''

    txt_msg = send_button.click(fn=ChatAgent.respond,
                                    inputs=[chatbot, input_txt],
                                    outputs=[input_txt,
                                                chatbot],
                                    queue=False).then(lambda: gr.Textbox(interactive=True),
                                                        None, [input_txt], queue=False)
    
    
    print("here")



# Launch Gradio App
demo.launch()