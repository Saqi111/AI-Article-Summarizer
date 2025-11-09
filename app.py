import gradio as gr
from transformers import pipeline

# 1. Load the summarization pipeline from Hugging Face.
# This model is downloaded and cached the first time the app runs.
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# 2. Define the core function to perform the summarization.
def summarize_text(article):
    """
    This function takes an article as a string and returns its summary.
    It includes checks for empty or very short articles.
    """
    # Check if the input article is empty.
    if not article:
        return "Please paste an article in the box above before submitting."
    
    # Check if the article is long enough to be summarized (e.g., more than 50 words).
    if len(article.split()) > 50:
        # Generate the summary using the loaded pipeline.
        summary = summarizer(article, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    else:
        # Return a message for articles that are too short.
        return "The article is too short to summarize. Please provide an article with more than 50 words."

# 3. Create the Gradio web interface for the application.
interface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(
        lines=15, 
        placeholder="Paste your long article here..."
    ),
    outputs=gr.Textbox(
        label="Summary"
    ),
    title="AI Article Summarizer",
    description="This tool uses an AI model to summarize long articles into a short, easy-to-read text. Just paste your article and click 'Submit' to see the magic!",
    allow_flagging='never',
    examples=[
        ["The history of the internet began with the development of electronic computers in the 1950s. Initial concepts of wide area networking originated in several computer science laboratories in the United States, United Kingdom, and France. The US Department of Defense awarded contracts as early as the 1960s, including for the development of the ARPANET project, directed by Robert Taylor and managed by Lawrence Roberts. The first message was sent over the ARPANET in 1969 from computer science Professor Leonard Kleinrock's laboratory at University of California, Los Angeles (UCLA) to the second network node at Stanford Research Institute (SRI)."]
    ]
)

# 4. Launch the web application.
interface.launch()
