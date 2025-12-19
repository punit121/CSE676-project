import json
import re
import os

import streamlit as st
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, pipeline
import torch
from torch import nn

class SmolVLMClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.vision_model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        for name, param in self.vision_model.named_parameters():
            if 'layer.23' not in name and 'layer.22' not in name and 'layer.21' not in name:
                param.requires_grad = False

        hidden_size = self.vision_model.config.text_config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        ).half()

        self.num_labels = num_labels

    def forward(self, input_ids, pixel_values, attention_mask=None):
        outputs = self.vision_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states[-1]
        pooled = hidden_states.mean(dim=1)
        pooled = pooled.half()

        logits = self.classifier(pooled)
        return logits


@st.cache_resource
def load_smolvlm_with_checkpoint(checkpoint_path: str = None):
    model_name = "HuggingFaceTB/SmolVLM-Instruct"
    num_labels = 6
    
    st.info(f"Loading SmolVLM from {model_name}")
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            st.success(f"Loading fine-tuned checkpoint from {checkpoint_path}")
            model = SmolVLMClassifier(model_name, num_labels)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint)
            model.eval()
            return model, processor, "classifier"
        except Exception as e:
            st.warning(f"Failed to load checkpoint: {e}. Falling back to base model.")
    
    st.warning("Using base SmolVLM generative model as fallback")
    pipe = pipeline(
        task="image-text-to-text",
        model=model_name,
    )
    return pipe, None, "generative"


LABELS = ["true", "satire", "fake_news", "false_connection", "misleading", "manipulated"]
LABEL_DESCRIPTIONS = {
    "true": "Factually accurate news",
    "satire": "Satirical or humorous content",
    "fake_news": "Completely fabricated information",
    "false_connection": "Misleading headline/image connection",
    "misleading": "Partially true but misleading",
    "manipulated": "Digitally altered or manipulated content"
}


st.set_page_config(page_title="SmolVLM Fake News Detector", layout="centered")
st.title("SmolVLM Fake News Detector")
st.write("Upload a news post image and provide its caption. SmolVLM will analyze both modalities and classify the post into one of 6 categories.")

with st.sidebar:
    st.header("Model Configuration")
    
    use_checkpoint = st.checkbox("Use fine-tuned checkpoint", value=True)
    
    if use_checkpoint:
        checkpoint_path = st.text_input(
            "Checkpoint path",
            value="/content/smolvlm_checkpoints/best_model.pt",
            help="Path to the .pt checkpoint file from training"
        )
    else:
        checkpoint_path = None
    
    st.divider()
    st.subheader("Model Info")
    st.write("Base Model: SmolVLM-Instruct")
    st.write("Classes: 6")
    st.write("Architecture: Vision-Language + Classifier Head")

model_or_pipe, processor, model_type = load_smolvlm_with_checkpoint(checkpoint_path if use_checkpoint else None)

if model_type == "classifier":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_or_pipe = model_or_pipe.to(device)
    st.success(f"Model loaded successfully. Running on {device}")
else:
    st.success("Generative model loaded successfully")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

caption = st.text_area(
    "Post text / caption",
    placeholder="Type or paste the text that appears with this image",
    height=120,
)

run_button = st.button("Analyze with SmolVLM")


@torch.no_grad()
def predict_with_classifier(model, processor, image, text, device):
    words = text.split()[:15]
    text = ' '.join(words)
    
    prompt = f"Classify: {text}"
    text_with_image = f"<image>{prompt}"
    
    inputs = processor(
        text=[text_with_image],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    
    input_ids = inputs['input_ids'].to(device)
    pixel_values = inputs['pixel_values'].to(device)
    attention_mask = inputs.get('attention_mask')
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    logits = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask
    )
    
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_idx = torch.argmax(probs, dim=-1).item()
    confidence = probs[0, pred_idx].item()
    all_probs = probs[0].cpu().numpy()
    
    return pred_idx, confidence, all_probs


def predict_with_generative(pipe, image, text):
    system_instruction = (
        "You are a fake-news detection assistant. "
        "Given a social media/news post, classify it into one of these labels: "
        "true, satire, fake_news, false_connection, misleading, manipulated. "
        "Return ONLY the label name, nothing else."
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": system_instruction + "\n\nPost text: " + text + "\n\nLabel:"},
            ],
        }
    ]
    
    outputs = pipe(
        text=messages,
        images=[image],
        max_new_tokens=32,
        return_full_text=False,
    )
    
    generated = outputs[0]["generated_text"].strip().lower()
    
    for label in LABELS:
        if label in generated:
            return label, 0.5
    
    return "unknown", 0.0


if run_button:
    if uploaded_file is None:
        st.error("Please upload an image")
    elif not caption.strip():
        st.error("Please enter the post text / caption")
    else:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Input Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Input Text")
            st.text_area("Caption", value=caption, height=200, disabled=True)

        st.markdown("---")
        st.subheader("SmolVLM Analysis")
        
        with st.spinner("Analyzing with SmolVLM"):
            try:
                if model_type == "classifier":
                    pred_idx, confidence, all_probs = predict_with_classifier(
                        model_or_pipe, processor, image, caption, device
                    )
                    predicted_label = LABELS[pred_idx]
                    
                    st.markdown("### Prediction")
                    
                    result_col1, result_col2 = st.columns([2, 1])
                    
                    with result_col1:
                        st.markdown(f"## {predicted_label.upper()}")
                        st.markdown(f"*{LABEL_DESCRIPTIONS[predicted_label]}*")
                    
                    with result_col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    st.markdown("---")
                    st.markdown("### Probability Distribution")
                    
                    prob_data = {
                        "Class": LABELS,
                        "Probability": all_probs
                    }
                    st.bar_chart(prob_data, x="Class", y="Probability")
                    
                    with st.expander("Detailed Probabilities"):
                        for i, label in enumerate(LABELS):
                            prob = all_probs[i]
                            st.write(f"{label}: {prob:.4f} ({prob*100:.2f}%)")
                
                else:
                    predicted_label, confidence = predict_with_generative(
                        model_or_pipe, image, caption
                    )
                    
                    st.markdown("### Prediction")
                    st.markdown(f"## {predicted_label.upper()}")
                    if predicted_label in LABEL_DESCRIPTIONS:
                        st.markdown(f"*{LABEL_DESCRIPTIONS[predicted_label]}*")
                    
                    if confidence > 0:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
            except Exception as e:
                st.error(f"Error during inference: {str(e)}")
                st.exception(e)

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>SmolVLM Fine-tuned for Fake News Detection</p>
    <p>Model: HuggingFaceTB/SmolVLM-Instruct + Classification Head</p>
    </div>
    """,
    unsafe_allow_html=True
)