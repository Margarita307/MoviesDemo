import streamlit as st
import base64
import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import plotly.express as px
from random import randrange

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
BG_IMAGE = "background.jpg"
BG_IMAGE_EXT = "jpg"

def main():
    st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{BG_IMAGE_EXT};base64,{base64.b64encode(open(BG_IMAGE, "rb").read()).decode()})
    }}
    .title {{
        color:#f5f5dc;
        font-weight:600;
        font-size:40px;
        text-shadow: 1px 1px 2px black, 0 0 1em red;
    }}
    .white_text {{
        color:#fafafa;
        font-weight:400;
        font-size:20px;
        text-shadow: 0px 0px 3px black, 0 0 1em red;
    }}
    /* White text containers */
    .element-container:nth-child(3) > div:nth-child(1) > div:nth-child(1),
    .element-container:nth-child(6) > div:nth-child(1) > div:nth-child(1) {{
        background-color: #27151566;
        border-radius: 4px;
        padding-left: 15px;
    }}
    /* I'm feeling lucky button */
    .element-container:nth-child(5) > div:nth-child(1) > button:nth-child(1) {{
        color: #FFF;
        letter-spacing: 1px;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.9);
        background: #434343 none repeat scroll 0% 0%;
        border: 1px solid #242424;
        box-shadow:
            0 1px 0 rgba(255, 255, 255, 0.25) inset,
            0 0 0 rgba(0, 0, 0, 0.5) inset,
            0 1.25rem 0 rgba(255, 255, 255, 0.08) inset,
            0 -1.25rem 1.25rem rgba(0, 0, 0, 0.3) inset,
            0 1.25rem 1.25rem rgba(255, 255, 255, 0.1) inset;
        transition: all 0.2s linear 0s;
        margin-bottom:40px;        
    }}
    /* details */
    div:nth-child(7) > div:nth-child(1) > ul:nth-child(1) > li:nth-child(1) > div:nth-child(1) {{
        color: #FFF;
        text-shadow: 0px 0px 3px black, 0 0 1em red;
        padding-left: 15px;
        text-transform: uppercase;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )   

    st.markdown('<p class="title">Movie genre recognition demo</p>', unsafe_allow_html=True)
    grid_search, multilabel_binarizer = get_grid_search_and_multilabel_binarizer()
    random_inputs = read_test_data()
    st.markdown('<div class="white_text" style="margin-bottom:-40px;">Enter citation from your favourite movie or choose at random</div>', unsafe_allow_html=True)
    placeholder = st.empty()
    input = placeholder.text_input("", )
    if st.button("I'm feeling lucky"):
        input = placeholder.text_input("", value=random_inputs[randrange(len(random_inputs))])
    if input:
        predict(grid_search, multilabel_binarizer, input)

def predict(grid_search, multilabel_binarizer, input):
    input = clean_x(input)
    input = pd.Series(input)
    vect = grid_search.best_estimator_.named_steps['vect']
    input = vect.transform(input)
    tfidf = grid_search.best_estimator_.named_steps['tfidf']
    input = tfidf.transform(input)
    clf = grid_search.best_estimator_.named_steps['clf']
    decision_function_output = clf.decision_function(input)
    sizes, lables = get_pie_sizes_and_lables(multilabel_binarizer, decision_function_output)
    print(lables)
    main_genres = []
    add_genres = []
    reply_message = ''
    for i in range(len(sizes)):
        if sizes[i] >= 0.4:
            main_genres.append(lables[i])
        else:
            add_genres.append(lables[i])
    if len(main_genres) > 0:
        if len(add_genres) > 0:
            reply_message = "This is " + ' and '.join(main_genres) + " with the elements of " + ' and '.join(add_genres) + '.'
        else:
            reply_message = "This is " + ' and '.join(main_genres) + '.'
    elif len(add_genres) > 0:
        reply_message = "This is the movie with the elements of " + ' and '.join(add_genres) + '.'
    else:
         reply_message = "Oops! It seems that we are unable to recognize the movie. Please try another citation."
    st.markdown('<div class="white_text">' + reply_message + '</div>', unsafe_allow_html=True)
    if len(sizes) > 1:
        placeholder = st.empty()
        with placeholder.beta_expander("See details"):
            show_pie_chart(sizes, lables)

def show_pie_chart(sizes, lables):
    fig = px.pie(sizes, values=sizes, names=lables)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend = dict(font = dict(size = 20, color = "#FFF"))
    )
    #fig, ax = plt.subplots()
    #ax.pie(sizes, labels=lables, autopct='%1.1f%%', startangle=90)
    #ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.plotly_chart(fig)

def get_pie_sizes_and_lables(multilabel_binarizer, decision_function_output):
    has_class = decision_function_output > 0
    lables = np.array([multilabel_binarizer.classes_])
    return decision_function_output[has_class]/decision_function_output[has_class].sum(), lables[has_class]

@st.cache(suppress_st_warning=True, show_spinner=False)
def get_grid_search_and_multilabel_binarizer():
    x, y, multilabel_binarizer = read_train_data()
    grid_search = train(x, y)
    return grid_search, multilabel_binarizer

def train(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(SGDClassifier())),
    ])
    parameters = {
        'vect__max_df': ([0.9]),
        'vect__max_features': ([8000]),
        'vect__ngram_range': ([(1, 1)]),
        'tfidf__use_idf': ([False]),
        'tfidf__norm': (['l2'])
    }
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=3)
    grid_search.fit(x_train, y_train)
    print(grid_search.best_params_)
    grid_search.refit
    y_pred = grid_search.predict(x_test)
    y_pred = (y_pred >= 0.6).astype(int)
    print(f1_score(y_test, y_pred, average="micro"))
    return grid_search

def read_train_data():
    data = pd.read_csv(TRAIN_FILE)
    data = data.groupby('movie').agg({'genres':', '.join,'dialogue':' '.join})
    x = process_x(data['dialogue'])
    y, multilabel_binarizer = process_y(data['genres'])
    return x, y, multilabel_binarizer

@st.cache(suppress_st_warning=True, show_spinner=False)
def read_test_data():
    data = pd.read_csv(TEST_FILE)
    return data['dialogue']

def process_x(text):
    nltk.download('stopwords')
    return text.apply(lambda x: clean_x(x))

def process_y(text):
    text = text.apply(lambda x: clean_y(x))
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(text)
    return multilabel_binarizer.transform(text), multilabel_binarizer

def clean_x(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stop_words.add('u')
    stop_words.add('br')
    tokens = [w for w in tokens if not w in stop_words]
    stemmer = SnowballStemmer("english")
    stems = []
    for t in tokens:    
        stems.append(stemmer.stem(t))
    return ' '.join(stems)

def clean_y(text):
    text = text[1:-1]
    text = re.sub("\[", "", text)
    text = re.sub("\]", "", text)
    text = re.sub("u'", "", text)
    text = re.sub("\'", "", text)
    return text.split(', ')

if __name__ == "__main__":
    main()

    #beta-expander
    #желтый стиль на кнопку
    #