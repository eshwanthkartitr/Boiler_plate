import gradio as gr

def demo_function(name):
    return f"Hello, {name if name else 'Developer'}! The OpenEnv Hackathon Demo is running successfully with the new updates!"

if __name__ == "__main__":
    print("Launching Gradio demo...")
    demo = gr.Interface(
        fn=demo_function, 
        inputs=gr.Textbox(label="Enter your name", placeholder="Name..."), 
        outputs=gr.Textbox(label="Message"),
        title="OpenEnv Hackathon Submission Demo (Updated v2 ✨)",
        description="A demo for your Hugging Face Space. This version has been updated to confirm your recent changes are now live!"
    )
    demo.launch()