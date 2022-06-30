import os
import spacy
import streamlit as st
import pandas as pd

# import numpy as np

import joblib


########################################

# !python -m spacy download en_core_web_sm -qq
nlp = spacy.load("en_core_web_sm")
# nlp.add_pipe("language_detector")

(dictionary, lda_model) = joblib.load(os.path.join("data", "lda.pipeline"))


def preprocessing(text, except_words=[]):

    # suppression des majuscules
    text = text.lower()

    # suppression des espaces au début et à la fin des textes
    text = text.strip()

    # tokenisation
    tokens = nlp(text)

    # récupère les lemmes après suppression des stopwords, de la ponctuation
    # des espaces et des adverbes et de ce qui n'est pas en anglais
    tokens = [
        token.lemma_
        for token in tokens
        if not token.is_stop
        # and doc._.language == 'en'
        # and doc._.language_score > 0.7
        and token.is_alpha
        and token.pos_
        not in [
            "ADV",
            "AUX",
            "CONJ",
            "CCONJ",
            "DET",
            "PART",
            "PRON",
            "PROPN",
            "PUNCT",
            "SCONJ",
            "SPACE",
            "SYM",
        ]
        and token.lemma_ not in except_words
    ]

    return tokens
    # return tokens if len(tokens) > 1 else "FILTERED"


def predict(texts):

    print("predict:", texts)
    print("predict:", type(texts))
    input_df = pd.DataFrame(texts, columns=["text"])
    print("01 OK")
    input_df["lemmas"] = input_df.text.apply(preprocessing)
    print("02 OK")
    input_bow = [dictionary.doc2bow(doc) for doc in input_df.lemmas]
    print("03 OK")
    input_pred = lda_model[input_bow]
    print("04 OK")


    for i in range(len(input_pred)):
        print_txt = input_df.text.iloc[i].replace('\n', ' ')
        scores = pd.DataFrame(input_pred[i], columns=['index','score']).set_index('index')

        st.write(f"---  \n#### Input #{i+1}")
        st.write(f"##### Texte avant traitement:  \n> {print_txt}")
        st.write(f"##### Texte après traitement:  \n> {input_df.lemmas.iloc[i]}")

        for j, score in enumerate(scores.sort_values('score', ascending=False).iterrows()):
            st.write(f"##### Sujet #{j+1}:  \n> {score[1][0]*100:.2f}% : {sujets[score[0]]}")

########################################


st.set_page_config(
        page_title="Démo Avis Resto",
        layout="centered",  # center | wide
        )

st.write(
    """
        # Démonstration des nouvelles fonctionnalités de collaboration pour "Avis Resto"
        ## Topic Modelling
        Dans cette première section, nous allons faire la démonstration de l'utilisation de notre modèle de topic modelling.
        """
)

# df = pd.DataFrame([1, 2, 3, 4, 1, 2, 3, 4, 5])
# st.line_chart(df)

sujets = {
        0: "Le sujet A",
        1: "Le sujet B",
        2: "Le sujet C",
        }

st.write("---  \n### Methode 1: saisie manuelle")
txt = st.text_area("Veuillez saisir (en anglais) une review négative dont vous aimeriez connaitre le sujet")


if txt is not None and txt != "":
    predict([txt])


st.write("---  \n### Methode 2: traitement en serie")
uploaded_file = st.file_uploader(
    "Choisissez un fichier TXT contenant une review par ligne.",
    type=['txt']
)
if uploaded_file is not None:
    # To read file as bytes:
    texts = []
    for line in uploaded_file:
        #st.write(line)
        texts.append(str(line))

    print(texts)
    print(type(texts))
    predict(texts)

st.write("## Classification des images utilisateur")

uploaded_files = st.file_uploader(
    "Choissisez une ou plusieurs images à analyser", accept_multiple_files=True
)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    # st.write("filename:", uploaded_file.name)
    # st.write(bytes_data)
    st.image(bytes_data)
    st.write("CLASSIFICATION:", "[pred text]")


with st.form("my-form", clear_on_submit=True):
    uploaded_files = st.file_uploader(
        "Choissisez une ou plusieurs images à analyser", accept_multiple_files=True
    )
    submitted = st.form_submit_button("UPLOAD!")

    if submitted and uploaded_files is not None:
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            # st.write("filename:", uploaded_file.name)
            # st.write(bytes_data)
            st.image(bytes_data)
            st.write("CLASSIFICATION:", "[pred text]")
