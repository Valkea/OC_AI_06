import streamlit as st
# import pandas as pd
# import numpy as np

st.write("""
        # Démonstration des nouvelles fonctionnalités de collaboration pour "Avis Resto"
        ## Topic Modelling
        Dans cette première section, nous allons faire la démonstration de l'utilisation de notre modèle de topic modelling.
        """)

# df = pd.DataFrame([1, 2, 3, 4, 1, 2, 3, 4, 5])
# st.line_chart(df)

st.write("### Methode 1: saisie manuelle")
txt = st.text_area("Veuillez saisir (en anglais) une review négative dont vous aimeriez connaitre le sujet", "")
print(txt)

st.write('Le sujet principal est:', txt[:10])
st.write('La répartition des sujets est:', "blop")

st.write("### Methode 2: traitement en serie")
uploaded_file = st.file_uploader("Choisissez un fichier TXT contenant une review par ligne.")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # st.write(bytes_data)

st.write("## Classification des images utilisateur")

uploaded_files = st.file_uploader("Choissisez une ou plusieurs images à analyser", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    # st.write("filename:", uploaded_file.name)
    # st.write(bytes_data)
    st.image(bytes_data)
    st.write("CLASSIFICATION:", "[pred text]")


with st.form("my-form", clear_on_submit=True):
    uploaded_files = st.file_uploader("Choissisez une ou plusieurs images à analyser", accept_multiple_files=True)
    submitted = st.form_submit_button("UPLOAD!")

    if submitted and uploaded_files is not None:
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            # st.write("filename:", uploaded_file.name)
            # st.write(bytes_data)
            st.image(bytes_data)
            st.write("CLASSIFICATION:", "[pred text]")
