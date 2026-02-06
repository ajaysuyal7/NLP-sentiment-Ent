import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# -----------------------------
# Backend URL
# -----------------------------
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("🧠 YouTube Comment Sentiment Analyzer")

st.write("Paste multiple comments (one per line):")

# -----------------------------
# Text Input
# -----------------------------
comments_text = st.text_area(
    "Comments",
    height=200,
    placeholder="This video is very good\nGood explanation but audio is bad\nWorst audio quality"
)

# -----------------------------
# Analyze Button
# -----------------------------
if st.button("Analyze"):
    if not comments_text.strip():
        st.warning("Please enter at least one comment.")
    else:
        comments_list = comments_text.strip().split("\n")

        payload = {
            "comments": [
                {"text": comment, "timestamp": f"00:{i:02d}"}
                for i, comment in enumerate(comments_list)
            ]
        }

        # -----------------------------
        # Call Sentiment API
        # -----------------------------
        with st.spinner("Analyzing sentiment..."):
            sentiment_response = requests.post(
                f"{API_URL}/predict", json=payload
            )

        if sentiment_response.status_code == 200:
            results = sentiment_response.json()

            st.subheader("📊 Sentiment Results")

            for r in results:
                sentiment_map = {
                    2: "Negative 😞",
                    0: "Neutral 😐",
                    1: "Positive 😊"
                }
                sentiment_label = sentiment_map.get(r['sentiment'])
                st.write(f"**{r['comment']}** → {sentiment_label}")
        else:
            st.error("Error from sentiment API")

        # -----------------------------
        # Call WordCloud API
        # -----------------------------
        with st.spinner("Generating word cloud..."):
            wc_response = requests.post(
                f"{API_URL}/Word-Cloud", json=payload
            )

        if wc_response.status_code == 200:
            st.subheader("☁️ Word Cloud")
            image = Image.open(BytesIO(wc_response.content))
            st.image(image, width=800)
        else:
            st.error(f"Error generating word cloud{wc_response.text}")

