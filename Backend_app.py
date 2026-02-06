from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import re
import pickle
import nltk

from io import BytesIO
from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Download NLTK resources once
# -----------------------------
nltk.download("stopwords")
nltk.download("wordnet")

#define loading model
def load_model(model_path, vectorizer_path):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
      
        return model, vectorizer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#loading model
model, vectorizer = load_model(
    "./lgbm_model.pkl", 
"./tfidf_vectorizer.pkl")

# Preprocess the comment
def preprocess_comment(comment:str) -> str :
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()
        # Remove trailing and leading whitespaces
        comment = comment.strip()
        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)
        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment


# -----------------------------
# Schemas
# -----------------------------
class Comment(BaseModel):
    text: str
    timestamp: str

class CommentRequest(BaseModel):
    comments: List[Comment]


#Define app
app = FastAPI(title="Sentiment Analysis API")

@app.get("/")
def read_root():
    return "Welcome to the Sentiment Analysis of comments"

#prediction endpoint
@app.post('/predict')
def predict_with_timestamps(request:CommentRequest):
    if not request.comments:
        raise HTTPException(status_code=400, detail="No comments provided")
    
    try:
        comments = [item.text for item in request.comments]
        timestamps = [item.timestamp for item in request.comments]

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Make predictions
        predictions = model.predict(transformed_comments)
        
        response = [
            {"comment": comment,
             "sentiment": int(sentiment),
              "timestamp": timestamp} 
              for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
              ]

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#word cloud endpoint

@app.post('/Word-Cloud')
def word_cloud(request:CommentRequest):
    if not request.comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    all_word=[]

    for comment in request.comments:
        processed_comment = preprocess_comment(comment.text)
        all_word.extend(processed_comment.split())
    
    if not all_word:
        raise HTTPException(status_code=400, detail="No valid words after preprocessing")
    
    word_freq = Counter(all_word)
    
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate_from_frequencies(word_freq)
    
    img = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(img, format="png")
    plt.close()
    img.seek(0)
    
    return StreamingResponse(img, media_type="image/png")
    
