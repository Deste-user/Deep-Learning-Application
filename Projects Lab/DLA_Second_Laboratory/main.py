import torch
import clip
from PIL import Image
import gradio as gr
import os
import pickle
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np

DATASET_PATH = "jxie/flickr8k"
DATASET_FOLDER = "./DATA/dataset"
EMBEDDINGS_FOLDER = "./DATA/embeddings"
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_FOLDER, "db.pkl")

def load_model(device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    translation_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-it-en")
    translation_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-it-en").to(device)
    return model, preprocess, translation_tokenizer, translation_model

def traduce_text(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    generated_tokens = translation_model.generate(**inputs)
    decoded_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return decoded_text


def create_embeddings(model, preprocess,device):
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            return pickle.load(f)

    print("Download the Dataset from Hugging Face!")
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    dataset = load_dataset(DATASET_PATH, split="train", cache_dir=DATASET_FOLDER)

    print("Create All Embeddings!")
    os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)

    db_features = []
    image_list = []
    model.eval()
    with torch.no_grad():
        for item in tqdm(dataset):
            img_pil = item["image"]
            image_input = preprocess(img_pil).unsqueeze(0).to(device)

            feat = model.encode_image(image_input)
            feat /= feat.norm(dim=-1, keepdim=True)
            db_features.append(feat.cpu())
            image_list.append(img_pil)

    db= {"embeddings": torch.cat(db_features), "images": image_list}

    with open(EMBEDDINGS_FILE,"wb") as f:
        pickle.dump(db, f)
    return db

def retrival_image(text, num_imgs):
    trad_text = traduce_text(text)

    text_tokenized = clip.tokenize([trad_text]).to(device)
    with torch.no_grad():
        feat_text = model.encode_text(text_tokenized)
        feat_text /= feat_text.norm(dim=-1, keepdim=True)

    feature_imgs = db["embeddings"].to(device)
    imgs = db["images"]

    similarities = (feat_text @ feature_imgs.T).squeeze(0)

    values, index = torch.topk(similarities, k=int(num_imgs))

    output = []
    for i in index:
        output.append(imgs[i])
    return [db["images"][i] for i in index]


with gr.Blocks() as app:
    gr.Markdown("# CLIP Retrievial System")
    with gr.Column():
        with gr.Row():
            txt_input = gr.Textbox(label="What are you looking for?")
            placeholder="Es. Un cucciolo che gioca nella neve..."
            num_input = gr.Slider(value=2, minimum=1, maximum=10, step=1, label="Number of Images")
            lines=2
        gr.Column(scale=1)    
        confirm_button = gr.Button("Confirm" ,size="sm", scale=1)
        gr.Column(scale=1)

    output_gallery = gr.Gallery(label="Results", columns=4, height="auto", show_label=False)

    confirm_button.click(fn=retrival_image, inputs = [txt_input,num_input], outputs=output_gallery, api_name="print_hello")

if __name__ == "__main__":
    device= "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, tokenizer, translation_model = load_model(device)
    db = create_embeddings(model, preprocess,device)
    app.launch()




