from flask import Flask, request, render_template
import pandas as pd
import re
import random
import os
import gdown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

MODEL_PATH = 'model.pkl'
FILE_ID = '1AYblw86kMxhfSBG7QwJWbjuHDI4ahwlx'
DOWNLOAD_URL = f'https://drive.google.com/uc?id={FILE_ID}'

if not os.path.exists(MODEL_PATH):
    print("Downloading model.pkl from Google Drive...")
    gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)

df = pd.read_csv("cars_ds_final_with_strengths.csv").drop_duplicates()


df = df[['Model', 'strengths', 'Ex-Showroom_Price']].dropna()


def parse_price(p):
    try:
        return int(str(p).replace("Rs.", "").replace(",", "").strip())
    except:
        return random.randint(500000, 2500000)

df['price'] = df['Ex-Showroom_Price'].apply(parse_price)


df['rating'] = [round(random.uniform(3.5, 5.0), 1) for _ in range(len(df))]


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

df['cleaned'] = df['strengths'].apply(clean_text)


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['cleaned'])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/aboutus")
def about():
    return render_template("aboutus.html")

@app.route("/contactus")
def contact():
    return render_template("contactus.html")

@app.route("/Explore")
def explore():
    return render_template("Explore.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form.get("strengths", "")

    if not user_input:
        return render_template("Explore.html", error="Please enter your car preferences.")

    cleaned_input = clean_text(user_input)
    input_vec = vectorizer.transform([cleaned_input])
    similarity = cosine_similarity(input_vec, tfidf_matrix).flatten()

    top_indices = similarity.argsort()[::-1][:8]
    top_cars = df.iloc[top_indices][['Model', 'strengths', 'price', 'rating']]

    return render_template("Explore.html", cars=top_cars.to_dict(orient='records'))

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=10000)
