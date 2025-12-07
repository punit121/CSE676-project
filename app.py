import json
import re

import streamlit as st
from PIL import Image
from transformers import pipeline

@st.cache_resource
def load_smolvlm_pipeline():
    """
    Load an instruction-tuned SmolVLM VLM as an image-text-to-text pipeline.

    You can replace the model id with:
      - your fine-tuned checkpoint (local path or HF repo), or
      - a larger SmolVLM variant.
    """
    model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
    pipe = pipeline(
        task="image-text-to-text",
        model=model_id,
        # device_map="auto",            # uncomment if using GPU
        # torch_dtype="auto",           # optional
    )
    return pipe


pipe = load_smolvlm_pipeline()


# ---------------------------------------------------------
st.set_page_config(page_title="SmolVLM Fake News Demo", layout="centered")
st.title("📰🔍 SmolVLM Fake News Detector")
st.write(
    """
Upload a **news post image** and provide its **caption/text**.  
SmolVLM will look at both and classify the post as one of:

`true, satire, fake_news, false_connection, misleading, manipulated`
"""
)

uploaded_file = st.file_uploader(
    "Upload an image (JPEG/PNG)", type=["jpg", "jpeg", "png"]
)

caption = st.text_area(
    "Post text / caption",
    placeholder="Type or paste the text that appears with this image...",
    height=120,
)

run_button = st.button("Analyze with SmolVLM 🚀")


LABELS = ["true", "satire", "fake_news", "false_connection", "misleading", "manipulated"]


def build_messages(caption_text: str):
    """
    Build the chat-style messages compatible with image-text-to-text pipeline.
    """
    system_instruction = (
        "You are a fake-news detection assistant. "
        "Given a social media/news post (image + text), classify it into one of these labels:\n"
        " - true\n"
        " - satire\n"
        " - fake_news\n"
        " - false_connection\n"
        " - misleading\n"
        " - manipulated\n\n"
        "Return a short JSON object ONLY, with keys: 'label', 'confidence', 'reason'.\n"
        "Example:\n"
        "{'label': 'fake_news', 'confidence': 0.82, 'reason': 'short explanation here'}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # the pipeline will inject the actual image
                {
                    "type": "text",
                    "text": (
                        system_instruction
                        + "\n\nPost text:\n"
                        + caption_text
                        + "\n\nNow return the JSON."
                    ),
                },
            ],
        }
    ]
    return messages


def parse_json_like(text: str):
    """
    Try to parse SmolVLM output as JSON or JSON-like dict.
    Fall back to a heuristic if needed.
    """
    # Replace single quotes with double quotes to help json.loads
    cleaned = text.strip()

    # Try to find a {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    cleaned_json = cleaned.replace("'", '"')

    try:
        data = json.loads(cleaned_json)
        return data
    except Exception:
        # Fallback: try to extract a label from known options
        label_found = None
        lowered = text.lower()
        for l in LABELS:
            if l in lowered:
                label_found = l
                break
        return {
            "label": label_found or "unknown",
            "confidence": None,
            "reason": text,
        }



if run_button:
    if uploaded_file is None:
        st.error("Please upload an image.")
    elif not caption.strip():
        st.error("Please enter the post text / caption.")
    else:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")

        # Display input image
        st.subheader("Input Image")
        st.image(image, use_column_width=True)

        st.subheader("SmolVLM Output")
        with st.spinner("Running SmolVLM..."):
            messages = build_messages(caption)
            # Pipeline expects messages + images
            outputs = pipe(
                text=messages,
                images=[image],
                max_new_tokens=128,
                return_full_text=False,  # only generated answer
            )
            generated = outputs[0]["generated_text"]

        # Raw generated text
        with st.expander("Raw model output"):
            st.code(generated)

        # Parsed result
        parsed = parse_json_like(generated)
        label = parsed.get("label", "unknown")
        confidence = parsed.get("confidence", None)
        reason = parsed.get("reason", "")

        # Nicely formatted result card
        st.markdown("---")
        st.subheader("Prediction")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Label:** `{label}`")
        with col2:
            if confidence is not None:
                try:
                    conf_val = float(confidence)
                    st.markdown(f"**Confidence:** `{conf_val:.2f}`")
                except Exception:
                    st.markdown(f"**Confidence:** `{confidence}`")
            else:
                st.markdown("**Confidence:** `N/A`")

        if reason:
            st.markdown("**Model's reasoning:**")
            st.write(reason)

