from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from pathlib import Path
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0"
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "saved_summary_model"

# -----------------------------
# Load Model
# -----------------------------
print("Loading model...")

tokenizer = T5Tokenizer.from_pretrained(str(MODEL_PATH))
model = T5ForConditionalGeneration.from_pretrained(str(MODEL_PATH))

# -----------------------------
# Device
# -----------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

model.to(device)
model.eval()

# -----------------------------
# Templates
# -----------------------------
templates = Jinja2Templates(directory=str(BASE_DIR))


class DialogueInput(BaseModel):
    dialogue: str


# -----------------------------
# Clean Text
# -----------------------------
def clean_data(text: str) -> str:
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip()

    return text


# -----------------------------
# Summarization
# -----------------------------
def summarize_dialogue(dialogue: str) -> str:

    dialogue = clean_data(dialogue)

    if not dialogue:
        return "Please enter some text."

    # T5 summarization prefix
    dialogue = "summarize: " + dialogue

    inputs = tokenizer(
        dialogue,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            min_length=10,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    summary = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return summary.strip()


# -----------------------------
# API
# -----------------------------
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):

    try:

        summary = summarize_dialogue(dialogue_input.dialogue)

        print("SUMMARY:", summary)

        return {
            "summary": summary
        }

    except Exception as e:

        print("ERROR:", str(e))

        return {
            "summary": "",
            "error": str(e)
        }


# -----------------------------
# Home Page
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )