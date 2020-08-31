from flask import Flask, render_template
import os
import torch

from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired

from transformers import DistilBertTokenizer, DistilBertModel

from model_funcs import *

'''
We load the model at the top of the app.
This means it will only get loaded into memory once on the server when we
deploy it, rather than being loaded every time we want to make a prediction.
'''

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

model = DistilBertModel.from_pretrained('distilbert-base-uncased',
                                        # Whether the model returns all hidden-states.
                                        output_hidden_states=True,
                                        )

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()


class TextForm(FlaskForm):
    string_field = StringField('text_field', validators=[DataRequired()])


app = Flask(__name__, static_url_path='/static')
# endow our app with a secret key so we can use FlaskForms
app.config['SECRET_KEY'] = os.urandom(32)


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    text_form = TextForm()
    text = None
    ranked_synonyms = None

    if text_form.validate_on_submit():
        text = text_form.string_field.data
        sentence = 'The news reports of our current ear can be scary.'

        with torch.no_grad():
            ranked_synonyms, ranked_scores = get_ranked_synonyms(
                model, tokenizer, sentence, text)

    return render_template('index.html', form=text_form, text=text, synonym=ranked_synonyms)
    # return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)  # port=os.getenv('PORT',5000)
