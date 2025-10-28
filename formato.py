import streamlit as st
import joblib

st.title('K-Means Clustering')
st.write('This app performs K-Means clustering on a dataset.')

model = joblib.load('kmeans_model.joblib')
scaler = joblib.load('scaler.joblib')

st.write('K-Means model loaded successfully.')

categorias = {
    "Neurológicos": [
        "mareo", "dolor de cabeza", "convulsiones", "pérdida de sensación",
        "alteración de la memoria", "debilidad", "parestesia", "somnolencia", "hablar", "nerv", "sensor", "focal", "delirio"
    ],
    "Respiratorios": [
        "tos", "dificultad para respirar", "dolor agudo en el pecho", "sibilancias",
        "fiebre", "congestión en el pecho", "tos con esputo", "apnea", "pulmon", "asma", "faring", "nariz", "sinus"
    ],
    "Digestivos": [
        "vómitos", "náuseas", "diarrea", "dolor abdominal agudo", "ictericia",
        "flatulencia", "acidez", "dolor abdominal bajo", "abdominal", "estómago", "heces", "reflujo", "esófago", "hígado", "intestino", "digestivo"
    ],
    "Urinarios y reproductivos": [
        "dolor al orinar", "micción frecuente", "flujo vaginal", "picazón vaginal",
        "dolor vaginal", "sequedad vaginal", "dolor testicular", "orina", "vagina", "útero", "pene", "testículo", "vejiga", "menstru", "embarazo", "prostata", "sexual", "uretra", "próstata"
    ],
    "Musculoesqueléticos": [
        "dolor muscular", "dolor de espalda", "dolor de cuello", "dolor de hombro",
        "rigidez o tirantez del cuello", "debilidad del brazo", "calambres o espasmos de espalda",
        "rigidez en todas partes", "calambres musculares, contracturas o espasmos",
        "calambres o espasmos lumbares", "masa o bulto en la espalda",
        "rigidez o tensión en las piernas", "dolor de costilla", "dolor en las articulaciones",
        "rigidez o tensión muscular", "dolor en la ingle", "calambres o espasmos en el codo",
        "rigidez o tirantez del cuello", "dolor de encías", "rigidez o tirantez de la cadera",
        "debilidad del brazo", "calambres o espasmos en el cuello", "problemas de postura",
        "encías sangrantes", "dolor en las encías", "dolor de mandíbula",
        "debilidad muscular", "hinchazón de las articulaciones", "debilidad del pie o del dedo del pie",
        "calambres o espasmos en las manos o los dedos", "rigidez o tirantez de la espalda",
        "bulto o masa en la muñeca", "dolor de piel", "rigidez o tirantez en la parte baja de la espalda",
        "tartamudear o tartamudear", "bulto sobre la mandíbula", "debilidad de la cadera",
        "hinchazón de espalda", "rigidez o tirantez del tobillo", "debilidad del tobillo", "debilidad del cuello", "hombro", "espalda", "pierna", "rodilla", "músculo", "dolor", "articulación", "codo", "cadera"
    ],
    "Cutáneos": ["piel", "erupción", "picazón", "acné", "mancha", "lunar", "quemadura", "labio", "uñas", "cuero cabelludo", "cabelludo"],
    "Otros": [] # symptoms that don't match any group
}
agrupados = {cat: [] for cat in categorias.keys()}





with st.form("input_form"):
    st.write("Entre los sintomas que usted tiene :")
    feature1 = st.number_input("dolor ocular")
    feature2 = st.number_input("síntomas oculares")
    feature3 = st.number_input("lagrimeo")
    feature4 = st.number_input("visión disminuida")
    feature5 = st.number_input("sensación de cuerpo extraño en el ojo")
    feature6 = st.number_input("picazón ocular")
    feature7 = st.number_input("depresión")
    feature8 = st.number_input("piel de aspecto anormal")
    feature9 = st.number_input("conducta hostil")
    feature10 = st.number_input("síntomas depresivos o psicóticos")
    feature11= st.number_input("enrojecimiento ocular")
    feature12= st.number_input("delirios o alucinaciones")
    feature13= st.number_input("lesión cutánea")
    feature14= st.number_input("hinchazón de la piel")
    feature15 = st.number_input("ira excesiva")







    submitted = st.form_submit_button("Submit")

    if submitted:
        input_data = [[feature1,feature2, feature3,feature4,
                       feature5,feature6,feature7,]]
        
        scaled_data = scaler.transform(input_data)
        cluster = model.predict(scaled_data)
       # print(cluster)
        mapped_cluster = MAPPING.get(cluster[0], 'Unknown Cluster')
        st.write(f'The predicted cluster is: {mapped_cluster}: {cluster[0]}')



