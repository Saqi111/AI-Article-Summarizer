import gradio as gr
from transformers import pipeline

# Yeh line AI model ko Hugging Face se download karti hai (sirf pehli baar)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(article):
    # Yeh function article leta hai aur summary wapas karta hai
    if len(article.split()) > 50:  # Agar article 50 lafzon se lamba ho to hi summary banayein
        summary = summarizer(article, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    else:
        return "Article bohat chota hai. Baraye meharbani, 50 se zyada lafzon ka article paste karein."

# Gradio Interface
# Yeh aapki website ka UI banata hai
interface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(lines=15, placeholder="Yahan apna lamba article paste karein..."),
    outputs=gr.Textbox(label="Khulasa (Summary)"),
    title="100% Free Article Summarizer",
    description="Yeh tool lambay articles ka khulasa banata hai. Sirf article paste karein aur 'Submit' par click karein.",
    allow_flagging='never',
    examples=[
        ["The history of the internet began with the development of electronic computers in the 1950s. Initial concepts of wide area networking originated in several computer science laboratories in the United States, United Kingdom, and France. The US Department of Defense awarded contracts as early as the 1960s, including for the development of the ARPANET project, directed by Robert Taylor and managed by Lawrence Roberts. The first message was sent over the ARPANET in 1969 from computer science Professor Leonard Kleinrock's laboratory at University of California, Los Angeles (UCLA) to the second network node at Stanford Research Institute (SRI)."]
    ]
)

# App ko launch karein
interface.launch()