from flask import Flask, render_template, request
from flask import render_template_string
from flask import redirect, url_for
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import re
import os
from flask_sqlalchemy import SQLAlchemy
import numpy as np
from test import perform_sentiment_analysis, sentimenthehe
import tensorflow as tf


delete_counter = 0

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///History.sqlite3'  # Use an SQLite database for simplicity
db = SQLAlchemy(app)

# Load delete counter from a file, if it exists
counter_file_path = 'delete_counter.txt'

if os.path.exists(counter_file_path):
    with open(counter_file_path, 'r') as file:
        delete_counter = int(file.read())

class History(db.Model):
    id_tweet = db.Column(db.Integer, primary_key=True, autoincrement=True)
    tweet = db.Column(db.String(255))
    positive = db.Column(db.Float)
    negative = db.Column(db.Float)
    neutral = db.Column(db.Float)
    tweet_type = db.Column(db.String(255))
    chatgpt = db.Column(db.Boolean)

# Load the sentiment analysis model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']

def determine_sentiment_type(scores):
    # Get the index of the maximum score
    max_index = np.argmax(scores)

    if max_index == 0:
        return 'negative'
    elif max_index == 1:
        return 'neutral'
    else:
        return 'positive'
    
# Define a function to detect mentions of "ChatGPT" in a tweet
def mentions_chatgpt(tweet):
    # Convert the tweet text to lowercase for case-insensitive matching
    tweet = tweet.lower()

    # Define a regular expression pattern to match variations of "ChatGPT"
    chatgpt_pattern = r'\b(chat\s?gpt|gpt-?3|openai)\b'

    # Use regular expressions to find matches in the tweet text
    matches = re.findall(chatgpt_pattern, tweet)

    # If there are matches, the tweet mentions "ChatGPT"
    return bool(matches)


@app.route('/')
def index():

    latest_tweets = History.query.limit(7).all()

    records = History.query.all()
    
    total_records = len(records)
    total_chatgpt_true = sum(1 for record in records if record.chatgpt)
    
    # Calculate percentages
    percent_chatgpt_true = (total_chatgpt_true / total_records) * 100

    # Filter records where chatgpt is true
    records_with_chatgpt = [record for record in records if record.chatgpt]

    # Calculate percentage of each tweet type
    tweet_types = [record.tweet_type for record in records_with_chatgpt]
    tweet_type_counts = {tweet_type: tweet_types.count(tweet_type) for tweet_type in set(tweet_types)}
    percent_tweet_types = {tweet_type: (count / total_chatgpt_true) * 100 for tweet_type, count in tweet_type_counts.items()}

    # Calculate percentage of each sentiment
    positive_tweets = sum(1 for record in records_with_chatgpt if record.tweet_type == 'positive')
    negative_tweets = sum(1 for record in records_with_chatgpt if record.tweet_type == 'negative')
    neutral_tweets = sum(1 for record in records_with_chatgpt if record.tweet_type == 'neutral')
    percent_positive = (positive_tweets / total_chatgpt_true) * 100
    percent_negative = (negative_tweets / total_chatgpt_true) * 100
    percent_neutral = (neutral_tweets / total_chatgpt_true) * 100
    
    records_with_chatgpt = History.query.filter_by(chatgpt=True).all()
    total_records_with_chatgpt = len(records_with_chatgpt)
    
    positiv = sum(1 for record in records if record.tweet_type == 'positive')
    negativ = sum(1 for record in records if record.tweet_type == 'negative')
    neutral = sum(1 for record in records if record.tweet_type == 'neutral')

    records = History.query.all()
    total_records = len(records)
    return render_template('index.html', 
                           total_records=total_records, 
                           delete_counter=delete_counter, 
                           total_records_with_chatgpt=total_records_with_chatgpt, 
                           percent_chatgpt_true=percent_chatgpt_true, 
                           percent_tweet_types=percent_tweet_types,
                           percent_positive=percent_positive,
                           percent_negative=percent_negative,
                           percent_neutral=percent_neutral,
                           positiv=positiv,
                           negativ=negativ,
                           neutral=neutral,
                           latest_tweets=latest_tweets)

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/about')
def about():
    return render_template('about.html')

model1 = tf.keras.models.load_model('cnnModel.h5')
model2 = tf.keras.models.load_model('lstmModel.h5')

@app.route('/analyze', methods=['POST'])
def analyze():
    tweet = request.form['tweet']

    sent1 = perform_sentiment_analysis([tweet],model1)
    sentiment1 = sentimenthehe(sent1)

    sent2 = perform_sentiment_analysis([tweet],model2)
    sentiment2 = sentimenthehe(sent2)
    

    # Preprocess tweet
    tweet_words = []
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    tweet_proc = " ".join(tweet_words)

    # Sentiment analysis
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    sentiment_results = [(label, score) for label, score in zip(labels, scores)]

    # Check if the tweet mentions "ChatGPT"
    mentions_chatgpt_flag = mentions_chatgpt(tweet)

    # Determine the sentiment type after calculating scores
    sentiment_type = determine_sentiment_type(scores)

    history_entry = History( 
                        tweet=tweet,
                        positive=scores[2],
                        negative=scores[0], 
                        neutral=scores[1],
                        tweet_type=sentiment_type,
                        chatgpt=mentions_chatgpt_flag

    )
    db.session.add(history_entry)
    db.session.commit()

    return render_template_string('''
        <title>Analysis Result</title>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
<link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Raleway">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
<style>
    html,
    body,
    h1,
    h2,
    h3,
    h4,
    h5 {
        font-family: "Raleway", sans-serif
    }
</style>
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script src="https://cdn.syncfusion.com/ej2/dist/ej2.min.js" type="text/javascript"></script>
<script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
<style>
    body {
        font-family: Arial, sans-serif;
        background-color: #f0f0f0;
        margin: 0;
        padding: 0;
    }

    .output {
        max-width: 800px;
        margin: 50px auto;
        padding: 20px;
        background-color: #fff;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        display: flex;
        flex-direction: column;
    }

    .resul-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        background-color: #fff;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }

    form {
        display: flex;
        flex-direction: column;
    }

    label {
        margin-bottom: 10px;
    }

    textarea {
        padding: 10px;
        margin-bottom: 15px;
    }

    input[type="submit"] {
        padding: 10px;
        background-color: #3498db;
        color: #fff;
        border: none;
        cursor: pointer;
    }

    input[type="submit"]:hover {
        background-color: #2980b9;
    }

    a {
        text-decoration: none;
        color: #3498db;
    }

    .Analysis {
        max-width: 100%;
        margin-bottom: 20px;
        padding: 20px;
        border: 1px solid #ddd;
        border-radius: 5px;
        background-color: #f9f9f9;
    }

    .Score {
        position: relative;
        left: 0px;
        max-width: 400px;
        margin: 0 auto;
        padding: 20px;
        border: 1px solid #ddd;
        border-radius: 5px;
        background-color: #f9f9f9;
    }

    .result-container {
        max-width: 350px;
        margin: 0 auto;
        padding: 20px;
        border: 1px solid #ddd;
        border-radius: 5px;
        background-color: #f9f9f9;
        left: auto;
    }

    .pie {
        max-width: 48%;
        /* Updated to 50% for side-by-side layout */
        margin: 0;
        padding: 20px;
        border: 1px solid #ddd;
        border-radius: 5px;
        background-color: #f9f9f9;
        box-sizing: border-box;
        /* Added to include padding and border in the width */
    }

    .sentiment-score {
        font-weight: bold;
    }

    .sentiment-bar {
        background-color: #3498db;
        color: white;
        display: inline-block;
        padding: 5px;
        margin-right: 5px;
        border-radius: 5px;
    }

    .sentiment-neutral {
        background-color: #f1c40f;
    }

    .sentiment-positive {
        background-color: #27ae60;
    }

    .sentiment-negative {
        background-color: #e74c3c;
    }

    .mention-chatgpt {
        font-weight: bold;
        color: #27ae60;
    }

    .no-mention-chatgpt {
        font-weight: bold;
        color: #e74c3c;
    }
</style>
<style>
    .container {
        width: 100%;
        box-sizing: border-box;
        /* Include padding and border in the width */
        padding: 20px;
    }

    .full-width {
        width: 100%;
        background-color: #f9f9f9;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
    }

    .text-below {
        margin-top: 10px;
    }

    .side-by-side-container {
        display: flex;
        margin-top: 10px;
    }

    .left-div {
        flex: 1;
        /* Takes up 2/3 of the space */
        background-color: #f9f9f9;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        box-sizing: border-box;
        /* Include padding and border in the width */
    }

    .right-div {
        flex: 1;
        /* Takes up 1/3 of the space */
        margin-left: 10px;
        /* Optional margin between the divs */
        background-color: #f9f9f9;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        box-sizing: border-box;
        /* Include padding and border in the width */
    }
                                  .additional-container {
        display: flex;
        margin-top: 10px;
    }

    .additional-div {
        flex: 1;
        /* Each div takes up equal space */
        margin-right: 10px;
        /* Optional margin between the divs */
        background-color: #f9f9f9;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        box-sizing: border-box;
        /* Include padding and border in the width */
    }
</style>
</head>

<body class="w3-light-grey">
    <!-- Navbar (sit on top) -->
    <div class="w3-top">
        <div class="w3-bar w3-white w3-wide w3-padding w3-card">
            <a href="/" class="w3-bar-item w3-button">Sentiment <b>ChatGPT</b></a>
            <!-- Float links to the right. Hide them on small screens -->
            <div class="w3-right w3-hide-small">
                <a href="/result" class="w3-bar-item w3-button">Analysis</a>
                <a href="/about" class="w3-bar-item w3-button">About</a>
                <a href="/display" class="w3-bar-item w3-button">Lists</a>
            </div>
        </div>
    </div>
    <br>
    <br>
    <div class="resul-container">
        <h1>Sentiment Analysis</h1>
        <form method="POST" action="/analyze">
            <label for="tweet">Enter a tweet:</label>
            <textarea name="tweet" id="tweet" rows="4" cols="50"></textarea>
            <br>
            <input type="submit" value="Analyze">
        </form>
        <div class="container">
            <div class="full-width">
                <h1>Analysis Result</h1>
                <p><strong>Tweet:</strong><br>{{ tweet }}</p>
            </div>
            <div class="text-below">
                <h2>Sentiment Scores:</h2>
            </div>
            <div class="side-by-side-container">
                <!-- Side-by-side divs -->
                <div class="left-div">
                    <ul>
                        {% for label, score in sentiment_results %}
                        <li>
                            <span class="sentiment-bar sentiment-{{ label|lower }}">
                                {{ label }}
                            </span>
                            <span class="sentiment-score">{{ score }}</span>
                        </li>

                        {% if label == "Negative" %}
                        {% with negative_score=score %}
                        <p>The negative score is: {{ negative_score }}</p>
                        <script>
                            var negativeScore = "{{ score }}";
                        </script>
                        {% endwith %}
                        {% endif %}

                        {% if label == "Neutral" %}
                        {% with neutral_score=score %}
                        <p>The neutral score is: {{ neutral_score }}</p>
                        <script>
                            var neutralScore = "{{ score }}";
                        </script>
                        {% endwith %}
                        {% endif %}

                        {% if label == "Positive" %}
                        {% with positive_score=score %}
                        <p>The positive score is: {{ positive_score }}</p>
                        <script>
                            var positiveScore = "{{ score }}";
                        </script>
                        {% endwith %}
                        {% endif %}

                        {% endfor %}
                    </ul>

                    <p>
                        {% if mentions_chatgpt %}
                        <span class="mention-chatgpt">Mentions ChatGPT</span>
                        {% else %}
                        <span class="no-mention-chatgpt">Doesn't Mention ChatGPT</span>
                        {% endif %}
                    </p>
                </div>
                <div class="right-div" id="piechart" style="width: 300px; height: 350px;"></div>
            </div>
                <div class="text-below">
                    <h2>Comparison:</h2>
                </div>
                <div class="additional-container">
                <!-- Three additional divs side by side -->
                <div class="additional-div">
                    <h2>CNN</h2>
                    <p>{{ sentiment1 }}</p>
                </div>
                <div class="additional-div">
                    <h2>LSTM</h2>
                    <p>{{ sentiment2 }}</p>
                </div>
            </div>
        </div>
    </div>
    <script type="text/javascript">
        google.charts.load('current', { 'packages': ['corechart'] });
        google.charts.setOnLoadCallback(drawChart);

        function drawChart() {

            var povSc = parseFloat(positiveScore);
            var negSc = parseFloat(negativeScore);
            var neuSc = parseFloat(neutralScore);
            var data = google.visualization.arrayToDataTable([
                ['Label', 'Score'],
                ['Positive', povSc],
                ['Negative', negSc],
                ['Neutral', neuSc]
            ]);

            var options = {
                title: 'Sentiment Result',
                colors: ['#27ae60', '#e74c3c', '#f1c40f']
            };

            var chart = new google.visualization.PieChart(document.getElementById('piechart'));

            chart.draw(data, options);
        }
    </script>
</body>
    ''', tweet=tweet, sentiment_results=sentiment_results, mentions_chatgpt=mentions_chatgpt_flag, sentiment1=sentiment1, sentiment2=sentiment2)
#, sentiment1=sentiment1

@app.route('/display')
def display_records():
    # Query the database to retrieve all records from the History table
    records = History.query.all()

    # Render a template and pass the records to it
    return render_template('display.html', records=records)


@app.route('/delete/<int:id_tweet>', methods=['POST'])
def delete_record(id_tweet):
    global delete_counter
    record_to_delete = History.query.get_or_404(id_tweet)

    # Delete the record from the database
    db.session.delete(record_to_delete)
    db.session.commit()

    delete_counter += 1

    # Save the updated counter to the file
    with open(counter_file_path, 'w') as file:
        file.write(str(delete_counter))
        
    return redirect(url_for('display_records'))
    


if __name__ == '__main__':
 with app.app_context():
    db.create_all()
    app.run(debug=True)
