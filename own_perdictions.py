from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

app = Flask(__name__)

# Load the saved model
loaded_model = pickle.load(open('./trained_model.sav', 'rb'))

# Load the vectorizer used for converting textual data into numerical data during training
vectorizer = pickle.load(open('./tfidf_vectorizer.sav', 'rb'))

pattern = re.compile('[^a-zA-Z]')
english_stopwords = stopwords.words('english')
port_stemmer = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub(pattern, ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stemmer.stem(word) for word in stemmed_content if not word in english_stopwords]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':  # Ensure the method is POST
        new_text = request.form['new_text']
        if new_text == "":
            prediction = "No statement provided"
        else:
            new_text_tfidf = vectorizer.transform([stemming(new_text)])
            prediction = loaded_model.predict(new_text_tfidf)
            prediction = "Good" if prediction == 1 else "Bad"
        return render_template('result.html', prediction=prediction)

    else:
        return "Method Not Allowed"

if __name__ == '__main__':
    app.run(debug=True)
