from flask import Flask, render_template, request

import summarizer
from summarizer import summarize_text_pagerank


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():

    summary = ''
    if request.method == 'POST':
        text = request.form['text']
        summary,summary_len,orginal_text = summarizer.summarize_text_pagerank(text)

    return render_template('index.html', summary=summary,summary_length = summary_len,originalText= orginal_text)

if __name__ == '__main__':
    app.run(debug=True)