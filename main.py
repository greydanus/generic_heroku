# Let's do the basic app configuration first:
from flask import Flask, render_template
import os

app = Flask(__name__, static_url_path='/static')
# endow our app with a secret key so we can use FlaskForms
app.config['SECRET_KEY'] = os.urandom(32)


# Now let's do application-specific imports and construct the demo
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired

import torch
from transformers import DistilBertTokenizer, DistilBertModel

from model_funcs import *

'''
We load the model at the top of the app.
This means it will only get loaded into memory once on the server when we
deploy it, rather than being loaded every time we want to make a prediction.
'''

# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# model = DistilBertModel.from_pretrained('distilbert-base-uncased',
#                                         # Whether the model returns all hidden-states.
#                                         output_hidden_states=True,
#                                         )

# Put the model in "evaluation" mode, meaning feed-forward operation.
# model.eval()


class ThesaurusInput(FlaskForm):
    sentence = StringField('Your sentence:', validators=[DataRequired()])
    word = StringField('The word you want to replace:', validators=[DataRequired()])


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    thesaurus_input = ThesaurusInput()
    synonyms_string = None

    if thesaurus_input.validate_on_submit():
        word = thesaurus_input.word.data
        sentence = thesaurus_input.sentence.data

        sentence = 'The news reports of our current era can be scary.' if sentence is None else sentence
        # word = 'scary'

        # with torch.no_grad():
        #     ranked_synonyms, ranked_scores = get_ranked_synonyms(
        #         model, tokenizer, sentence, text)
        synonyms = get_unranked_synonyms(word)
        synonyms_string = ','.join(synonyms[:5])

    return render_template('index.html', thesaurus_input=thesaurus_input, synonyms=synonyms_string)
    # return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)  # port=os.getenv('PORT',5000)
