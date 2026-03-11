from flask import Flask,render_template,request
import pickle

app = Flask(__name__)

model = pickle.load(open("model/sentiment_model.pkl","rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl","rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():

    review = request.form['review']

    review_vec = vectorizer.transform([review])

    prediction = model.predict(review_vec)

    if prediction[0] == 1:
        result = "Positive Review 😊"
    else:
        result = "Negative Review 😞"

    return render_template("result.html",prediction=result)

if __name__ == "__main__":
    app.run(debug=True)