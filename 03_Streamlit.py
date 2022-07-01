import os
import joblib

import spacy
import spacy_fastlang
import streamlit as st
import pandas as pd

##########################################
# Topic Modelling : functions & variables
##########################################

# Initialize the spacy nlp pipeline
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("language_detector")

# Load the LDA model and the associated dictionnary
(dictionary, lda_model) = joblib.load(os.path.join("data", "lda.pipeline"))
sujets = {
    0: "Le sujet A",
    1: "Le sujet B",
    2: "Le sujet C",
}

# Define required functions
def preprocessing(text, except_words=[]):
    """
    This function aims to prepare the provided text for the model.
    It processes only 1 text at a time and hence it needs to be called via myDF.apply(preprocessing)
    """

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
        and tokens._.language == "en"
        and tokens._.language_score > 0.7
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


def is_filtered_docs(lemmas):
    return len(lemmas) == 0


def predict(texts):

    input_df = pd.DataFrame(texts, columns=["text"])
    input_df["lemmas"] = input_df.text.apply(preprocessing)
    input_df["filtered"] = input_df.lemmas.apply(is_filtered_docs)
    input_bow = [dictionary.doc2bow(doc) for doc in input_df.lemmas]
    input_pred = lda_model[input_bow]

    for i in range(len(input_pred)):
        print_txt = input_df.text.iloc[i]
        # print_txt = input_df.text.iloc[i].replace('\\n', ' ')

        scores = pd.DataFrame(input_pred[i], columns=["index", "score"]).set_index(
            "index"
        )

        st.write(f"---  \n#### Input #{i+1}")
        st.write(f"##### Texte avant traitement:  \n> {print_txt}")
        if input_df.filtered.iloc[i]:
            st.write("##### Ce texte ne peut pas être traité...")
        else:
            st.write(f"##### Texte après traitement:  \n> {input_df.lemmas.iloc[i]}")

            for j, score in enumerate(
                scores.sort_values("score", ascending=False).iterrows()
            ):
                st.write(
                    f"##### Sujet #{j+1}:  \n> {score[1][0]*100:.2f}% : {sujets[score[0]]}"
                )


########################################
# Streamlit design
########################################

st.set_page_config(
    page_title="Démo Avis Resto",
    page_icon="🍔",
    layout="centered",  # center | wide
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.google.com/help",
        "Report a bug": "https://www.google.com/bug",
        "About": "# This is a header. This is an *extremely* cool app!",
    },
)

st.title('Démonstration des nouvelles fonctionnalités de collaboration pour "Avis Resto"')
st.write("## Topic Modelling")
st.write("---  \n### Methode 1: saisie manuelle")
txt = st.text_area(
    "Veuillez saisir (en anglais) une review négative dont vous aimeriez connaitre le sujet"
)


if txt is not None and txt != "":
    predict([txt])


st.write("---  \n### Methode 2: traitement en serie")
uploaded_file = st.file_uploader(
    "Choisissez un fichier TXT contenant une review par ligne.", type=["txt"]
)
if uploaded_file is not None:
    # To read file as bytes:
    texts = []
    for line in uploaded_file:
        txt = line.decode("utf-8")
        texts.append(txt)

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
