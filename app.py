from flask import Flask,render_template,request,redirect
from helper import preprocessing,get_prediction,vectorization
from logger import logging
import os

app = Flask(__name__)

logging.info('Flask Server Started')

data = dict()
reviews = []
positive = 0
negative = 0

# Home route (GET)
@app.route("/", methods=["GET"])
def index():
    global reviews, positive, negative

    data['reviews'] = reviews
    data['positive'] = positive
    data['negative'] = negative

    logging.info("======== Open home Page =========")

    return render_template('index.html', data=data)

# Form submit (POST)
@app.route("/", methods=["POST"])
def my_post():
    global positive, negative, reviews

    text = request.form.get('text')

    logging.info(f'Text : {text}')

    # Preprocess
    preprocessed_text = preprocessing(text)
    logging.info(f'PreProcessed Text : {preprocessed_text}')

    # Vectorize
    vectorized_text = vectorization(preprocessed_text)
    logging.info(f'Vectorized Text : {vectorized_text}')

    # Predict
    prediction = get_prediction(vectorized_text)
    logging.info(f'Prediction : {prediction}')

    # Count results
    if prediction == 'negative':
        negative += 1
    else:
        positive += 1

    # Save review
    reviews.insert(0, text)

    return redirect("/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Heroku dynamic port
    app.run(host="0.0.0.0", port=port, debug=False)