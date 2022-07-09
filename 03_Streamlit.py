#!/usr/bin/env python
# coding: utf-8

import os
import math
import time
import joblib

import pandas as pd
import numpy as np

import streamlit as st
from streamlit_option_menu import option_menu

import spacy
import spacy_fastlang

import tflite_runtime.interpreter as tflite
from PIL import Image, ImageOps, ImageFilter  # , ImageDraw
import matplotlib.pyplot as plt
from matplotlib import gridspec
# from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16


##################################################
# Topic Modelling : functions & variables
##################################################

# Initialize the spacy nlp pipeline
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("language_detector")

# Load the LDA model and the associated dictionnary
(dictionary, lda_model, topic_labels) = joblib.load(
    os.path.join("data", "lda.pipeline")
)


# Define required functions
def preprocessing(text, except_words=[]):
    """
    This function aims to prepare the provided text for the model. The various
    steps will clean the text and eventually return the extracted lemmas.

    It processes only 1 text at a time and hence it needs to be called via
    myDF.apply(preprocessing) (because it is more efficient than nlp.pipe)

    Parameters
    ----------
    text: str
        The text to preprocess in order to extract the lemmas
    except_words : list
        The list of the potential extra words we dont want to see in the corpus

    Returns
    -------
    list:
        The list of the lemmas extrated from the provided text
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


def is_filtered_docs(lemmas):
    """
    This function returns a boolean indicating if the provided
    lemmas list can be used by the model or not.

    Parameters
    ----------
    text: str
        The lemmas list returned by the preoprecessing function.

    Returns
    -------
    boolean:
        A boolean indicating if this document can be used on the model.
    """
    return len(lemmas) == 0


def predict_topics(texts):
    """
    This function is a pipeline that takes string texts and process them
    to prepare the data and make prediction use the Topic-modelling model.

    Then finally, it calls a function to print the results

    Parameters
    ----------
    text: str
        The lemmas list returned by the preoprecessing function.
    """

    input_df = pd.DataFrame(texts, columns=["text"])
    input_df["lemmas"] = input_df.text.apply(preprocessing)
    input_df["filtered"] = input_df.lemmas.apply(is_filtered_docs)
    input_bow = [dictionary.doc2bow(doc) for doc in input_df.lemmas]
    input_pred = lda_model[input_bow]

    print_topics(input_pred, input_df)


def print_topics(preds, input_df):
    """
    This function prints the model predictions using the Streamlit functions

    Parameters
    ----------
    preds: array
        The prediction array returned by the model.
    input_df: pandas.DataFrame
        The dataframe containing the preprocessing transformations
    """

    for i in range(len(preds)):
        print_txt = input_df.text.iloc[i].replace("\n", " ")

        scores = pd.DataFrame(preds[i], columns=["index", "score"]).set_index("index")

        st.write(f"---  \n#### Input #{i+1}")
        st.write(f"##### Texte avant traitement:  \n> {print_txt}")
        if input_df.filtered.iloc[i]:
            st.write("##### ⚠️ Ce texte ne peut pas être traité...")
        else:
            st.write("##### Texte après traitement:")
            st.write(f"> {input_df.lemmas.iloc[i]}")

            for j, score in enumerate(
                scores.sort_values("score", ascending=False).iterrows()
            ):
                st.write(f"##### Sujet #{j+1}:")
                st.write(f"> {score[1][0]*100:.2f}% : {topic_labels[score[0]]}")


def get_top_id(row):
    """TODO"""

    max_id = None
    max_va = 0
    for topics in row:
        cur_id = topics[0]
        cur_va = topics[1]
        if cur_va > max_va:
            max_va = cur_va
            max_id = cur_id

    if math.isclose(max_va, 0.33333, rel_tol=1e-1):
        return "None"
    else:
        return f"{topic_labels[max_id]} ({max_va*100.0:.2f}%)"


##################################################
# Image Classification : functions & variables
##################################################

# --- Load TF-Lite model using an interpreter
interpreter = tflite.Interpreter(model_path="models/vgg16_clf10.tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

categories = ["drink", "food", "inside", "menu", "outside"]
categories_fr = ["boisson", "nourriture", "intérieur", "menu", "extérieur"]
# collected from validation_flow.class_indices.keys()


def preprocess_image_show(steps_show, steps_name):

    # print(img_path, photo.label.upper())

    fig = plt.figure(figsize=(20, 7), facecolor="lightgray")

    spec = gridspec.GridSpec(
        ncols=len(steps_show),
        nrows=2,
        width_ratios=[1] * len(steps_show),
        wspace=0.3,
        hspace=0.3,
        height_ratios=[5, 2],
    )

    for i, image in enumerate(steps_show):

        fig.add_subplot(spec[i])
        plt.title(steps_name[i])
        plt.imshow(image, cmap="gray", vmin=0, vmax=255)
        plt.axis("off")

        fig.add_subplot(spec[i + len(steps_show)])
        mat = np.array(image)
        plt.title("Histogramme de l'image")
        plt.hist(mat.flatten(), bins=range(256))
        plt.xlabel("Intensité")
        plt.ylabel("Nombre de pixels")

    # plt.show()
    st.pyplot(fig)


def preprocess_image(img, param_blur=2, newsize=(300, 300)):

    img = Image.open(img)

    # Blur
    blured_img = img.filter(ImageFilter.BoxBlur(param_blur))

    # Equalize
    equalized_img = ImageOps.equalize(blured_img)

    # Resize
    final_img = equalized_img.resize(newsize)

    steps_show = [img, blured_img, equalized_img, final_img]
    steps_name = ["source", "blured", "equalized", "resized"]
    preprocess_image_show(steps_show, steps_name)

    return blured_img, equalized_img, final_img


def predict_category(img):

    img = np.array(img, np.float32)
    # img = preprocess_input_vgg16(img)

    # Apply model
    interpreter.set_tensor(input_index, [img])
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    top_score = preds[0].max()
    top_label = categories_fr[preds[0].argmax()]
    return top_label, top_score, preds[0]


##################################################
# Streamlit design
##################################################


st.set_page_config(
    page_title="Démo Avis Resto",
    page_icon="🍔",
    layout="wide",  # center | wide
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.google.com/help",
        "Report a bug": "https://www.google.com/bug",
        "About": "# This is a header. This is an *extremely* cool app!",
    },
)

st.title(
    'Démonstration des nouvelles fonctionnalités de collaboration pour "Avis Resto"'
)


def show_topic_modelling():

    # --- Single Topic Modelling ---
    st.write("### Methode 1: saisie manuelle")
    txt = st.text_area(
        "Veuillez saisir (en anglais) une review négative dont vous aimeriez connaitre le sujet"
    )

    if txt is not None and txt != "":
        predict_topics([txt])

    # --- Batch Topic Modelling ---

    st.write("### Methode 2: traitement en serie")
    uploaded_file = st.file_uploader(
        "Choisissez un fichier TXT contenant une review par ligne.", type=["txt", "csv"]
    )
    if uploaded_file is not None:
        # To read file as bytes:
        texts = []
        if uploaded_file.type == "text/csv":
            input_df = pd.read_csv(uploaded_file)
            print(input_df)
            columns = input_df.columns

            with st.spinner("Traitement..."):

                input_df["lemmas"] = input_df.text.apply(preprocessing)
                input_df["filtered"] = input_df.lemmas.apply(is_filtered_docs)
                input_bow = [dictionary.doc2bow(doc) for doc in input_df.lemmas]
                input_pred = pd.DataFrame(lda_model[input_bow])
                input_pred["main_topic"] = input_pred.apply(get_top_id, axis=1)
                # st.dataframe(input_df)
                # st.dataframe(input_pred)

                export_df = input_df[columns]
                export_df["main_topic"] = input_pred["main_topic"]
                st.dataframe(export_df)

                st.download_button(
                    "Télécharger ce nouveau jeu de données",
                    export_df.to_csv(index=False),
                    "file.csv",
                    "text/csv",
                    key="download-csv",
                )

        elif uploaded_file.type == "text/plain":
            for line in uploaded_file:
                txt = line.decode("utf-8")
                texts.append(txt)

            with st.spinner("Traitement..."):
                predict_topics(texts)


def show_image_classification():

    uploaded_files = st.file_uploader(
        "Choissisez une ou plusieurs images à analyser",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png"],
    )
    for i, uploaded_file in enumerate(uploaded_files):
        st.write(f"---  \n#### Input #{i+1}")
        t0 = time.time()
        bytes_data = uploaded_file.read()
        t1 = time.time()
        print(f"IMAGE CLASSIFICATION TIME > read bytes > {t1-t0:.4f}")
        blured_img, equalized_img, final_img = preprocess_image(
            uploaded_file, newsize=(224, 224)
        )
        st.image([bytes_data, final_img], width=350)
        t2 = time.time()
        print(f"IMAGE CLASSIFICATION TIME > preprocess_image > {t2-t1:.4f}")
        top_label, top_score, preds = predict_category(final_img)
        t3 = time.time()
        print(f"IMAGE CLASSIFICATION TIME > predict_category > {t3-t2:.4f}")
        st.write(
            f"CLASSIFICATION: <span style='color:Grey'>{[round(x,4) for x in preds]}</span> >>> <span style='color:Red'>{top_label.title()}</span> ({top_score*100:.2f}%)",
            unsafe_allow_html=True,
        )


# --- Side bar ---

with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Topic Modelling", "Image Classification"],
        icons=["newspaper", "camera"],
        default_index=0,
    )

if selected == "Topic Modelling":
    st.write("---  \n## Topic Modelling")
    show_topic_modelling()
else:
    st.write("---  \n## Classification des images utilisateur")
    show_image_classification()
