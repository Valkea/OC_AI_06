#!/usr/bin/env python
# coding: utf-8

import os
import math
import joblib
import pathlib

import pandas as pd
import numpy as np

import streamlit as st
from streamlit_option_menu import option_menu
from streamlit import components

import spacy
import spacy_fastlang

import tflite_runtime.interpreter as tflite
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
# from sklearn.cluster import KMeans
# from keras.models import load_model

import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import cv2
import seaborn as sns
import pyLDAvis.gensim_models

##################################################
# Topic Modelling : functions & variables
##################################################

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("language_detector")


@st.cache(allow_output_mutation=True)
def load_lda():
    print("LOADING the LDA model and the associated dictionnary")
    (dictionary, lda_model, topic_labels, corpus_bow) = joblib.load(
        os.path.join("models", "lda_vis.pipeline")
    )
    return dictionary, lda_model, topic_labels, corpus_bow


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
    """
    This function search for the most probable topic
    given the provided scores list.

    If the row was totally Filtered (most probably because
    it's a foreign language), the function returns None
    instead of the tuple containing the main topic and its
    score.

    Parameters
    ----------
    row: pd.Series
        The list of scores for the various topics

    Returns
    -------
    tuple:
        a tuple containing the main topic label along with it's score.
    """

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


@st.cache(allow_output_mutation=True)
def load_feature_extractor():
    print("LOADING feature extractor")

    CNN_feature_extractor = joblib.load(
        pathlib.Path("models", "feature_extractor_CNN.bin")
    )
    # CNN_feature_extractor = load_model(pathlib.Path("models", "feature_extractor_CNN.h5"))
    # CNN_feature_extractor.load_weights(pathlib.Path("models", "feature_extractor_CNN_weights.hdf5"))

    return CNN_feature_extractor


@st.cache(allow_output_mutation=True)
def load_CNN_tsne():
    print("LOADING CNN t-sne")

    (
        tsne_CNN_trained_data,
        tsne_CNN_trained_model,
        tsne_CNN_trained_labels,
    ) = joblib.load(pathlib.Path("models", "tsne_CNN_trained_dual.bin"))

    return tsne_CNN_trained_data, tsne_CNN_trained_model, tsne_CNN_trained_labels


@st.cache(allow_output_mutation=True)
def load_SIFT_tsne():
    print("LOADING SIFT t-sne")

    (
        tsne_SIFT_trained_data,
        tsne_SIFT_trained_model,
        tsne_SIFT_trained_labels,
    ) = joblib.load(pathlib.Path("models", "tsne_SIFT_dual.bin"))

    return tsne_SIFT_trained_data, tsne_SIFT_trained_model, tsne_SIFT_trained_labels


@st.cache
def load_CNN_classifier():
    print("LOADING CNN classifier")

    global CNN_classifier, input_index, output_index
    CNN_classifier = tflite.Interpreter(
        model_path=str(pathlib.Path("models", "vgg16_clf.tflite"))
    )
    CNN_classifier.allocate_tensors()
    input_index = CNN_classifier.get_input_details()[0]["index"]
    output_index = CNN_classifier.get_output_details()[0]["index"]

    return CNN_classifier, input_index, output_index


# --- Define labels
# collected from validation_flow.class_indices.keys()
categories = ["drink", "food", "inside", "menu", "outside"]
categories_fr = ["boisson", "nourriture", "intérieur", "menu", "extérieur"]


def preprocess_image_show(steps_show, steps_name):
    """
    This function shows the various images and associated histograms
    generated by the preprocessing pipeline.

    Parameters
    ----------
    steps_show: list
        The list images to display
    steps_name: list
        The list of the steps names to display next to the images
    """

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


def preprocess_image(img, param_blur=2, newsize=(300, 300), show_preprocess=False):
    """
    This function preprocesses the provided image in order
    to prepare it for the CNN model.

    Parameters
    ----------
    img: bytes_array
        The raw data of the image to preprocess
    param_blur: int
        The number of pixels to use on the blur filter
    newsize: tuple(int, int)
        The size of the final image
    show_preprocess: boolean
        A boolean that defines if we should show the complete pipeline or not

    Returns
    -------
    byte_array:
        the final image generated by the preprocessing pipeline
    """

    img = Image.open(img)

    # Blur
    blured_img = img.filter(ImageFilter.BoxBlur(param_blur))

    # Equalize
    equalized_img = ImageOps.equalize(blured_img)

    # Resize
    final_img = equalized_img.resize(newsize)

    if show_preprocess:
        steps_show = [img, blured_img, equalized_img, final_img]
        steps_name = ["source", "blured", "equalized", "resized"]
        preprocess_image_show(steps_show, steps_name)

    return final_img


def predict_category(img):
    """
    This function provide the model with a prepcocessed image
    and return the results (most probable label, best score, scores lists)

    Parameters
    ----------
    img: bytes_array
        The preprocessed image for which we want to predict the category

    Returns
    -------
    str:
        the label associated with the best prediction score
    float:
        the best score returned by the model with the provided image
    list:
        the scores as returned by the model for all categories
    """

    img = np.array(img, np.float32)
    img = preprocess_input_vgg16(img)

    # Apply model
    CNN_classifier.set_tensor(input_index, [img])
    CNN_classifier.invoke()
    preds = CNN_classifier.get_tensor(output_index)
    top_score = preds[0].max()
    top_label = categories_fr[preds[0].argmax()]
    return top_label, top_score, preds[0]


def plot_TNSE_with_new_points(
    model,
    old_data,
    new_data,
    labels=None,
    title="t-SNE",
    alpha=0.5,
    color_target="cluster",
):
    """
    This function plots the provided t-SNE model along with the old and new features.

    Parameters
    ----------
    model: openTSNE
        The fitted openTSNE model
    old_data: pd.DataFrame
        The dataset containing the data already transformed with the t-SNE model
    new_data: array
        The new features to transform using the model
    labels: list
        The list of optional labels for the t-SNE clusters
    color_target: str
        The name of the column that need to be used for the coloration
    alpha: float
        The opacity value for the old features on the plot
    title: str
        The title of the plot
    """

    cmap_ref = "nipy_spectral"
    new_data = model.transform(new_data)
    new_data = pd.DataFrame(new_data, columns=["D1", "D2"])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(
        old_data["D1"],
        old_data["D2"],
        c=old_data[color_target],
        s=50,
        cmap=cmap_ref,
        marker="+",
        alpha=alpha,
    )
    ax.scatter(new_data["D1"], new_data["D2"], s=50, color="r", marker="o")
    if labels:
        plt.legend(handles=scatter.legend_elements()[0], labels=labels)
    else:
        plt.colorbar(scatter)
    ax.set_title(title)
    ax.set_xlabel("Dimention 1")
    ax.set_ylabel("Dimention 2")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    h_space, w_space = 10, 5
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * (w + w_space) - w_space, rows * (h + h_space)))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        sub_img = Image.new("RGB", size=(w, h + h_space))
        sub_img.paste(img, box=(0, h_space))
        draw = ImageDraw.Draw(sub_img)
        txt = f"{i+1}"
        ts = draw.textlength(txt)
        draw.text(((w - ts) / 2, 0), txt, (255, 255, 255), align="center")
        grid.paste(sub_img, box=(i % cols * (w + w_space), i // cols * (h + h_space)))
    return grid


def preprocess_image_SIFT(img, num_clusters=95):

    root_num_top = 4
    sift = cv2.SIFT_create(nfeatures=250)  # patchSize is fixed in code (size=12σ×12σ)
    kmeans_SIFT, num_clusters = joblib.load(pathlib.Path("models", "kmeans_SIFT.bin"))

    mat = np.array(img)
    # queryKeypointsORB, queryDescriptorsORB = orb.detectAndCompute(mat,None)
    queryKeypointsSIFT, queryDescriptorsSIFT = sift.detectAndCompute(mat, None)

    # Get visual-words for the histogramm
    preds = pd.DataFrame(kmeans_SIFT.predict(queryDescriptorsSIFT))
    select = (
        pd.DataFrame(preds.value_counts(sort=False), columns=["count"])
        .reset_index()
        .rename(columns={0: "index"})
    )
    select.set_index("index", inplace=True)
    select = select.reindex(list(range(0, num_clusters)), fill_value=0)
    select_top = select.sort_values("count", ascending=False)[: root_num_top ** 2]

    fig = plt.figure(figsize=(20, 7), facecolor="lightgray")

    # draw only keypoints location,not size and orientation
    plt.subplot(1, 6, (1, 2))

    plt.title("SIFT descriptors")
    img_sift = cv2.drawKeypoints(
        mat, queryKeypointsSIFT, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    plt.imshow(img_sift)
    plt.axis("off")

    # draw visual-words histogram
    plt.subplot(1, 6, (3, 5))

    ax = sns.barplot(data=select.T)
    ax.bar_label(ax.containers[0])
    new_ticks = [i.get_text() for i in ax.get_xticklabels()]
    labels_modulo = 5
    plt.xticks(
        range(0, len(new_ticks), labels_modulo), new_ticks[::labels_modulo], rotation=0
    )
    plt.ylabel("Nombre d'occurences")
    plt.xlabel("Index des visual-words")
    plt.title(f"Histogramme")

    # draw visual-words
    grid_imgs = []
    for j in range(0, root_num_top ** 2):
        print('j:', j)
        index = select_top.index[j]

        # Select one of the multiple patches as visual-word representation
        id = preds[preds.values == index].sample(1, random_state=0).index[0]

        # Get the patch infos
        (x_s, y_s) = queryKeypointsSIFT[id].pt
        size_s = queryKeypointsSIFT[id].size
        angle_s = queryKeypointsSIFT[id].angle

        # SIFT Crop
        left_s = x_s - size_s
        right_s = x_s + size_s
        top_s = y_s - size_s
        bottom_s = y_s + size_s

        img_copy = img.copy().rotate(-angle_s, center=(x_s, y_s))
        img_crop_s = img_copy.crop((left_s, top_s, right_s, bottom_s)).resize(
            (30, 30), Image.Resampling.NEAREST
        )

        # Append to grid list
        grid_imgs.append(img_crop_s)

    plt.subplot(1, 6, 6)
    plt.title(f"TOP {root_num_top**2} visual-words")
    plt.imshow(image_grid(grid_imgs, rows=root_num_top, cols=root_num_top))
    plt.axis("off")

    # plt.show()
    st.pyplot(fig)
    st.dataframe(select.T)

    return select


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

# st.title(
#    'Démonstration des nouvelles fonctionnalités de collaboration pour "Avis Resto"  \n---'
# )


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
        type=["jpg", "jpeg"],
    )
    show_preprocess = st.checkbox("Afficher les étapes de pré-traitement", value=False)

    for i, uploaded_file in enumerate(uploaded_files):
        st.write(f"---  \n#### Input #{i+1}")
        bytes_data = uploaded_file.read()

        final_img = preprocess_image(uploaded_file, 2, (224, 224), show_preprocess)

        col1, col2, col3 = st.columns([1, 2, 1])

        if show_preprocess is False:
            with col2:
                st.image([bytes_data, final_img], width=350)

        top_label, top_score, preds = predict_category(final_img)

        with col2:
            st.write(
                f"CLASSIFICATION: <span style='color:Grey'>{[round(x,4) for x in preds]}</span> >>> <span style='color:Red'>{top_label.title()}</span> ({top_score*100:.2f}%)",
                unsafe_allow_html=True,
            )


def show_image_feature_extraction():
    uploaded_files = st.file_uploader(
        "Choissisez une ou plusieurs images à analyser",
        accept_multiple_files=True,
        type=["jpg", "jpeg"],
    )
    show_preprocess = st.checkbox("Afficher les étapes de pré-traitement", value=True)

    for i, uploaded_file in enumerate(uploaded_files):
        st.write(f"---  \n#### Input #{i+1}")
        bytes_data = uploaded_file.read()

        final_img = preprocess_image(uploaded_file, 2, (224, 224), show_preprocess)
        col1, col2, col3 = st.columns([1, 2, 1])

        if show_preprocess is False:
            with col2:
                st.image([bytes_data, final_img], width=350)

        pred_img = np.array(final_img, np.float32)
        pred_img = np.expand_dims(pred_img, axis=0)
        bags_of_visual_words_CNN = CNN_feature_extractor.predict(pred_img)

        plot_TNSE_with_new_points(
            tsne_CNN_trained_model,
            tsne_CNN_trained_data,
            bags_of_visual_words_CNN,
            labels=tsne_CNN_trained_labels,
            color_target="category",
            title="t-SNE des features extraites du CNN",
            alpha=0.75,
        )


def show_image_feature_extraction_SIFT():
    uploaded_files = st.file_uploader(
        "Choissisez une ou plusieurs images à analyser",
        accept_multiple_files=True,
        type=["jpg", "jpeg"],
    )
    show_preprocess = st.checkbox("Afficher les étapes de pré-traitement", value=True)

    for i, uploaded_file in enumerate(uploaded_files):
        st.write(f"---  \n#### Input #{i+1}")
        bytes_data = uploaded_file.read()

        final_img = preprocess_image(uploaded_file, 2, (224, 224), show_preprocess)
        col1, col2, col3 = st.columns([1, 2, 1])

        if show_preprocess is False:
            with col2:
                st.image([bytes_data, final_img], width=350)

        bovw = preprocess_image_SIFT(final_img)
        bovw = np.expand_dims(bovw, axis=0)

        plot_TNSE_with_new_points(
            tsne_SIFT_trained_model,
            tsne_SIFT_trained_data,
            bovw,
            labels=tsne_SIFT_trained_labels,
            color_target="category",
            title="t-SNE des features extraites du CNN",
            alpha=0.75,
        )

    # if uploaded_files:
    #     st.write("---")
    #     col1, col2, col3 = st.columns([1, 1, 1])
    #     with col2:
    #         st.write("#### Cependant, on peut voir sur le t-SNE ci-dessous qu'avec ce nombre de visual-words dans le BoVW il est difficile de distinguer les catégories...")
    #         img1 = Image.open(pathlib.Path("medias", "t-SNE-SIFT95.png"))
    #         st.image(img1)


def show_text_feature_extraction():
    option = st.selectbox(
        "Les wordclouds aux différentes étapes",
        (
            "Avant traitement",
            "Après tokenisation + filtrage + lemmatization",
            "Après suppression des extrêmes (en fréquence)",
            "Application d'une LDA sur le corpus préparé",
        ),
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    # st.write('You selected:', option)
    if option == "Avant traitement":

        img1 = Image.open(pathlib.Path("medias", "wordcloud1.png"))
        img2 = Image.open(pathlib.Path("medias", "stars1.png"))
        img3 = Image.open(pathlib.Path("medias", "stars2.png"))
        txt1 = """
        ### Nous avons commencé par faire une sélection de 10.000 documents dont les évaluations *(stars)* étaient de 1 ou 2.
        """

        with col2:
            st.image(img1)
            st.write(txt1)
            st.write("Répartition des notes pour les 150.346 documents d'origine")
            st.image(img2)
            st.write(
                "Répartition des notes après sélection de 10.000 documents au hasard"
            )
            st.image(img3)

    elif option == "Après tokenisation + filtrage + lemmatization":
        img1 = Image.open(pathlib.Path("medias", "wordcloud2.png"))
        txt1 = """
        ### Tokenisation
        > Cette étape consiste à découper les phrases ou documents en mot individuels ou en bloc de mots *(bigrammes, trigrammes, etc.)*. Se faisant, nous constituons un **corpus vocabulary** qui peut être utilisé de plusieurs façon *(Bags of Words, TF-IDF, Word2Vec...)*
        >#### Lors de cette étape nous avons également supprimé:
        > - les majuscules,
        > - les espaces en début et fin de texte,

        ### Filtrage des tokens
        > Cette étape consiste à supprimer les tokens qui n'ont pas de valeur ajouté pour notre futur modèle, donc des mots qui ne transportent pas d'information utile pour le projet.
        >#### Lors de cette étape nous avons donc supprimé:
        > - tout ce qui n'est pas détecté comme étant de l'anglais *(language & language_score)*,
        > - les stop-words *(is_stop)*,
        > - la ponctuation *(PUNCT)*,
        > - les espaces *(SPACE)*,
        > - les chiffres *(is_alpha)*,
        > - mais aussi les beaucoup d'autres tags peu utiles dans ce cas *(ADV, AUX, CONJ, CCONJ, DET, PART, PRON, PROPN, SCONJ, SYM)*.

        ### Lemmatisation
        > Cette étape consiste à chercher la raçine commune des mots en utilisant le contexte *(alors que le stemming n'utilise pas le contexte)*. Se faisant, nous rapprochons des mots qui pourraient sinon être considérés comme différents par nos algorithmes.
        """

        with col2:
            st.image(img1)
            st.write(txt1)

    elif option == "Après suppression des extrêmes (en fréquence)":
        img1 = Image.open(pathlib.Path("medias", "wordcloud3.png"))
        txt1 = """
        ### Filtrage des lemmes selon leur fréquence
        > - On a supprimé les mots qui apparaissent dans moins de 5 documents
        > - On a supprime les mots qui apparaissent dans plus de 50% des documents

        ### Préparation de trois représentations *numériques"* des documents
        > - Le `Corpus - Bag Of Word` qui contient une liste *(un BoW pour chaque document)* de *term vectors* qui indiquent la fréquence de chaque mot du *vocabulaire*.
        > - Le `Corpus - TF-IDF` qui est une version normalisée du Bag-Of-Words pour éviter que des mots sans importance mais fréquents ne prennent trop d'importance.
        > - Le `Corpus - Word2vec` associe un vecteur à chaque mot du *vocabulaire* au lieu d'une valeur de fréquence (normalisée ou non). Cett particularité permet de retrouver plus facilement des mots similaires *( Homme --> Chercheur | Femme --> ? --> Chercheuse)*.
        >
        > Ces trois réprésentations ont été utilisés pour essayer différentes variantes des algorithmes `Latent Dirichlet Allocation (LDA)` et `Negative Matrix Factorisation (NMF)` 
        """

        with col2:
            st.image(img1)
            st.write(txt1)

    elif option == "Application d'une LDA sur le corpus préparé":

        txt1 = """
        ### Latent Dirichlet Allocation *(LDA)*
        C'est une méthode non-supervisée générative vraiment efficace qui se base sur les hypothèses suivantes :
        - Chaque document du corpus est un ensemble de mots sans ordre (bag-of-words)
        - Chaque document *m* aborde un certain nombre de thèmes dans différentes proportions qui lui sont propres *p(θm)*
        - Chaque mot possède une distribution associée à chaque thème *p(ϕk)*. On peut ainsi représenter chaque thème par une probabilité sur chaque mot.
        - *z_n* représente le thème du mot *w_n*

        > ⚠️ "In fact, Blei (who developed LDA), points out in the introduction of the paper of 2003 (entitled "Latent Dirichlet Allocation") that LDA addresses the shortcomings of the TF-IDF model and leaves this approach behind. LSA is compeltely algebraic and generally (but not necessarily) uses a TF-IDF matrix, while LDA is a probabilistic model that tries to estimate probability distributions for topics in documents and words in topics. The weighting of TF-IDF is not necessary for this."
        >
        > ⚠️ Le modèle **TF-IDF peut améliorer les résultats d'un LDA** dans le cas d'un **nombre extrêmement important de documents**. Mais dans l'ensemble, le **Bag-Of-Words est plus approprié** pour le modèle LDA.
        """

        txt2 = """
        >### Les topics identifés sont:
        > - 1: "La qualité du service",
        > - 2: "La qualité des produits proposés",
        > - 3: "Une déception dans un établissement apprécié",
        """

        img1 = Image.open(pathlib.Path('medias', 'LDA_process.png'))

        with col2:
            st.write(txt1)
            st.image(img1)
            st.write(txt2)

        lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus_bow, dictionary)
        # pyLDAvis.display(lda_display)

        html_string = pyLDAvis.prepared_data_to_html(lda_display)
        components.v1.html(f"<body style='background-color:white'>{html_string}</body>", width=1300, height=800)


# --- Side bar ---

with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=[
            "TXT Feature Extraction",
            "Topic Modelling",
            "---",
            "SIFT Feature Extraction",
            "CNN Feature Extraction",
            "Image Classification",
        ],
        icons=["", "newspaper", "", "", "", "camera"],
        default_index=0,
    )

global dictionary, lda_model, topic_labels, corpus_bow

if selected == "Topic Modelling":
    st.write("## Topic Modelling")

    dictionary, lda_model, topic_labels, corpus_bow = load_lda()

    show_topic_modelling()

elif selected == "Image Classification":
    st.write("Classification des images utilisateur")

    global CNN_classifier, input_index, output_index
    CNN_classifier, input_index, output_index = load_CNN_classifier()

    show_image_classification()

elif selected == "SIFT Feature Extraction":
    st.write("## Extraction des features avec SIFT")

    global tsne_SIFT_trained_data, tsne_SIFT_trained_model, tsne_SIFT_trained_labels
    (
        tsne_SIFT_trained_data,
        tsne_SIFT_trained_model,
        tsne_SIFT_trained_labels,
    ) = load_SIFT_tsne()

    show_image_feature_extraction_SIFT()

elif selected == "CNN Feature Extraction":
    st.write("## Extraction des features avec le CNN")

    global CNN_feature_extractor
    CNN_feature_extractor = load_feature_extractor()

    global tsne_CNN_trained_data, tsne_CNN_trained_model, tsne_CNN_trained_labels
    (
        tsne_CNN_trained_data,
        tsne_CNN_trained_model,
        tsne_CNN_trained_labels,
    ) = load_CNN_tsne()

    show_image_feature_extraction()

else:
    st.write("## Préparation des documents")

    dictionary, lda_model, topic_labels, corpus_bow = load_lda()

    show_text_feature_extraction()
