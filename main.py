# Let's do the basic app configuration first:
from flask import Flask, render_template
import os

app = Flask(__name__, static_url_path='/static')
# endow our app with a secret key so we can use FlaskForms
app.config['SECRET_KEY'] = os.urandom(32)


# Now let's do application-specific imports and construct the demo
from flask_wtf import FlaskForm
from wtforms import StringField, BooleanField
from wtforms.validators import DataRequired

import torch

from model_funcs import *

'''
We load the model at the top of the app.
This means it will only get loaded into memory once on the server when we
deploy it, rather than being loaded every time we want to make a prediction.
'''

from transformers import AutoTokenizer, AutoModel
# See https://huggingface.co/google/bert_uncased_L-10_H-512_A-8
# Load the largest BERT-like model that will fit on a free heroku instance
tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-8_H-512_A-8")
model = AutoModel.from_pretrained("google/bert_uncased_L-8_H-512_A-8")

# from transformers import DistilBertTokenizer, DistilBertModel
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# model = DistilBertModel.from_pretrained('distilbert-base-uncased',
#                                         # Whether the model returns all hidden-states.
#                                         output_hidden_states=True,
#                                         )

# # Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()


class ThesaurusInput(FlaskForm):
    sentence = StringField('Your sentence: ', default="You are pretty great", validators=[DataRequired()])
    word = StringField('The word you want to replace: ', default="pretty", validators=[DataRequired()])
    use_deep = BooleanField('Deep thesaurus enabled: ', default="checked")

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def submit():
    thesaurus_input = ThesaurusInput()
    reg_synonyms_str = deep_synonyms_str = None

    if thesaurus_input.validate_on_submit():
        word = thesaurus_input.word.data
        sentence = thesaurus_input.sentence.data
        # Make sure it ends in a period. The BERT ranking works *much* with a period added.
        sentence = sentence + '.' if sentence[-1] != '.' else sentence

        # get regular synonyms
        reg_synonyms = get_unranked_synonyms(word)

        # get deep thesaurus synonyms
        if thesaurus_input.use_deep.data:
            with torch.no_grad():
                ranked_synonyms, ranked_scores = get_ranked_synonyms(
                    model, tokenizer, sentence, word)
                deep_synonyms = ranked_synonyms
        else:
            deep_synonyms = ['Deep synonyms not requested']
        
        # string formatting
        CHOP_AT = 50
        reg_synonyms, reg_synonyms_leftover = reg_synonyms[:CHOP_AT], reg_synonyms[CHOP_AT:]
        deep_synonyms, deep_synonyms_leftover = deep_synonyms[:CHOP_AT], deep_synonyms[CHOP_AT:]
        reg_synonyms_str = ', '.join(reg_synonyms) + ', ... (+{} more)'.format(len(reg_synonyms_leftover))
        deep_synonyms_str = ', '.join(deep_synonyms) + ', ... (+{} more)'.format(len(deep_synonyms_leftover))

    return render_template('index.html', thesaurus_input=thesaurus_input,
                            reg_synonyms=reg_synonyms_str,
                            deep_synonyms=deep_synonyms_str)

if __name__ == '__main__':
    app.run(debug=True)  # port=os.getenv('PORT',5000)


    

