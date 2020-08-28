from flask import Flask, render_template
import os

from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired

class TextForm(FlaskForm):
    string_field = StringField('text_field', validators=[DataRequired()])

app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = os.urandom(32)  # endow our app with a secret key so we can use FlaskForms

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    text_form = TextForm()
    text =  None

    if text_form.validate_on_submit():
        text = text_form.string_field.data

    return render_template('index.html', form=text_form, text=text)
    # return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv('PORT',5000))