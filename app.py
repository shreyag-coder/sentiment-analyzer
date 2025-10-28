# app.py
import os
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Review Sentiment + Aspect", page_icon="📝")
st.title("📝 Review Sentiment Classifier (+ tiny Aspect add-on)")

LOCAL_MODEL_DIR = "./sentiment_model"


@st.cache_resource
def load_sentiment_pipeline():
    if os.path.isdir(LOCAL_MODEL_DIR):
        try:
            tok = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
            mdl = AutoModelForSequenceClassification.from_pretrained(
                LOCAL_MODEL_DIR)
            return pipeline("sentiment-analysis", model=mdl, tokenizer=tok, device_map="auto")
        except Exception:
            pass
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device_map="auto")


@st.cache_resource
def load_zeroshot_pipeline():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device_map="auto")


clf = load_sentiment_pipeline()

st.subheader("Enter a product review")
text = st.text_area("Review text", height=140,
                    placeholder="e.g., Delivery was slow but quality is excellent.")

col1, col2 = st.columns(2)
with col1:
    aspect_mode = st.radio("Aspect tagging mode", [
                           "None", "Keyword match", "Zero-shot (MNLI)"], index=1)
with col2:
    default_labels = "delivery, price, quality, packaging, return, warranty"
    label_str = st.text_input(
        "Aspect labels (comma-separated)", value=default_labels)


def keyword_aspects(text, labels):
    text_low = text.lower()
    hits = []
    for lab in labels:
        lab2 = lab.strip().lower()
        if lab2 and lab2 in text_low:
            hits.append(lab.strip())
    return hits


if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    out = clf(text)[0]
    st.markdown(
        f"**Sentiment:** `{out['label']}`  |  **Score:** `{out['score']:.3f}`")

    labels = [s.strip() for s in label_str.split(",") if s.strip()]
    if aspect_mode == "Keyword match":
        hits = keyword_aspects(text, labels)
        st.markdown("**Aspects (keyword):** " +
                    (", ".join(hits) if hits else "_None_"))
    elif aspect_mode == "Zero-shot (MNLI)":
        zsc = load_zeroshot_pipeline()
        res = zsc(text, candidate_labels=labels, multi_label=True)
        keep = [(lab, score) for lab, score in zip(
            res["labels"], res["scores"]) if score >= 0.35]
        if keep:
            keep.sort(key=lambda x: x[1], reverse=True)
            st.markdown("**Aspects (zero-shot, ≥0.35):**")
            for lab, sc in keep:
                st.write(f"- {lab}: {sc:.2f}")
        else:
            st.markdown("_No aspects above threshold._")

st.caption(
    "Tip: Train first to use your local fine-tuned weights in ./sentiment_model")
