from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# Load trained model
classifier = pipeline("text-classification", model="query_classifier", tokenizer="query_classifier")

# Define queries
queries = [
    "Latest iPhone deals",
    "Where can I find Elon Musk's favorite books?",
    "Best budget gaming laptop",
    "Cristiano Ronaldo's diet plan",
    "Playstation 5",
    "Adidas size 5",
    "Which Adidas shoe would Andrew Garfield wear"
]

@app.get("/", response_class=HTMLResponse)
async def index():
    results = []
    for q in queries:
        result = classifier(q)[0]
        label = "Direct Embedding Search" if result["label"] == "LABEL_0" else "Keyword Extraction Needed"
        results.append((q, label))

    html = """
    <html>
        <head><title>Query Classification</title></head>
        <body>
            <h1>Query Classification Results</h1>
            <table border="1" cellpadding="10">
                <tr><th>Query</th><th>Classification</th></tr>
    """
    for q, label in results:
        html += f"<tr><td>{q}</td><td>{label}</td></tr>"

    html += """
            </table>
        </body>
    </html>
    """
    return html
