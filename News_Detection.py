from flask import Flask, request, render_template
import joblib
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

application = Flask(__name__)

# Load the trained model and vectorizer
stack_model = joblib.load("models/new_stack_model.joblib")
vectorizer_model = joblib.load("models/new_vectorizer.joblib")

# API Keys
FACT_CHECK_API_KEY = "AIzaSyDhOWPEQeBbb-0VwW6ncozyaNbQdlXoDDA"
NEWS_API_KEY = "8b0bba112f594fc2b5b98488845d722f"

@application.route("/")
def index():
    return render_template("index.html")

@application.route("/home")
def home_page():
    return render_template("home.html")

@application.route("/Check_for_news", methods=["POST"])
def check_news():
    if request.method == "POST":
        news_text = request.form["news"]

        # ML Prediction
        input_vector = vectorizer_model.transform([news_text])
        proba = stack_model.predict_proba(input_vector)[0][1]
        threshold = 0.8 
        confidence_score = proba
        prediction = 1 if proba > threshold else 0
        result = f"{'Real News' if prediction == 1 else 'Fake News'} (Confidence: {round(proba * 100, 2)}%)"

        # Google Fact Check
        fact_result = "Not Found"
        fact_link = None

        try:
            response = requests.get(
                "https://factchecktools.googleapis.com/v1alpha1/claims:search",
                params={"key": FACT_CHECK_API_KEY, "query": news_text, "languageCode": "en"}
            )
            data = response.json()
            if "claims" in data and len(data["claims"]) > 0:
                fact_result = "Found"
                claim_review = data["claims"][0].get("claimReview", [])
                if claim_review:
                    fact_link = claim_review[0].get("url")
        except Exception as e:
            print("Fact Check API error:", e)

        matched_article = None
        similarity_score = None
        newsapi_results = []
        newsapi_empty = False

        if fact_result == "Not Found":
            try:
                news_response = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": news_text,
                        "language": "en",
                        "pageSize": 20,
                        "sortBy": "relevancy",
                        "apiKey": NEWS_API_KEY
                    }
                )
                if news_response.status_code == 200:
                    articles = news_response.json().get("articles", [])
                    if not articles:
                        newsapi_empty = True
                    else:
                        formatted_articles = []
                        for a in articles:
                            raw_time = a.get("publishedAt", "")
                            try:
                                dt_object = datetime.strptime(raw_time, "%Y-%m-%dT%H:%M:%SZ")
                                formatted_time = dt_object.strftime("Published on: %B %d, %Y at %I:%M %p")
                            except:
                                formatted_time = "Time not available"
                            formatted_articles.append({
                                "title": a["title"],
                                "url": a["url"],
                                "description": a.get("description", ""),
                                "published": formatted_time,
                                "source": a.get("source", {}).get("name", "Unknown")
                            })

                        corpus = [news_text] + [a["title"] + ". " + a["description"] for a in formatted_articles]
                        tfidf_matrix = vectorizer_model.transform(corpus)
                        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
                        max_index = np.argmax(similarities)
                        similarity_score = similarities[max_index]
                        if similarity_score >= 0.35:
                            matched_article = formatted_articles[max_index]
                            newsapi_results = formatted_articles[:max_index] + formatted_articles[max_index + 1:]
                        else:
                            matched_article = None
                            similarity_score = None
                            newsapi_results = []
                            newsapi_empty = True
                else:
                    newsapi_empty = True
            except Exception as e:
                print("NewsAPI error:", e)
                newsapi_empty = True

        # Warnings
        high_confidence_unverified = (
            prediction == 1 and confidence_score > 0.90
            and fact_result == "Not Found"
            and matched_article is None and newsapi_empty
        )

        show_general_warning = (
            fact_result == "Not Found"
            and matched_article is None
            and newsapi_empty
        )

        return render_template(
            "result.html",
            news=news_text,
            prediction=result,
            fact_result=fact_result,
            fact_link=fact_link,
            matched_article=matched_article,
            similarity_score=similarity_score,
            newsapi_results=newsapi_results,
            newsapi_empty=newsapi_empty,
            high_confidence_unverified=high_confidence_unverified,
            show_general_warning=show_general_warning
        )

if __name__ == "__main__":
    application.run(host="127.0.0.1", port=5000, debug=False)
