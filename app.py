
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and the symptom groupings
# Assuming the model was saved as 'random_forest_model.joblib'
# Assuming the symptom groupings were saved or are available as 'agrupados'
# If the model and agrupados are not saved, they should be re-created
# For this example, let's assume 'agrupados' is available from previous execution
# and the model will be loaded if saved, otherwise re-trained (not ideal for deployment)

# In a real application, you would load the pre-trained model like this:
try:
    modelo = joblib.load('modelo_enfermedades.pkl')
except FileNotFoundError:
    st.error("Error: Trained model not found. Please train the model first.")
    st.stop() # Stop the app if the model is not available

# Load the data to get symptom names
try:
    df = pd.read_excel("Enfermedades y sintomas.xlsx")
    sintomas = df.columns[:-1].tolist()
except FileNotFoundError:
    st.error("Error: Data file not found. Please ensure 'Enfermedades y sintomas.xlsx' is in the correct location.")
    st.stop()

# Recreate the 'agrupados' dictionary with symptom categories
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

for s in sintomas:
    asignado = False
    for cat, palabras in categorias.items():
        if any(palabra.lower() in s.lower() for palabra in palabras):
            agrupados[cat].append(s)
            asignado = True
            break
    if not asignado:
        agrupados["Otros"].append(s)


st.title("🩺 Formulario de Síntomas Médicos")

# Select category
categoria = st.selectbox("Selecciona un grupo de síntomas", list(agrupados.keys()))

# Select symptoms
sintomas_seleccionados = st.multiselect(
    "Selecciona tus síntomas (mínimo 3):",
    options=agrupados[categoria]
)

# Verify minimum quantity
if len(sintomas_seleccionados) < 3:
    st.warning("⚠️ Debes seleccionar al menos **3 síntomas** para continuar.")
else:
    st.success(f"✅ Has seleccionado {len(sintomas_seleccionados)} síntomas.")
    st.write("Síntomas seleccionados:", sintomas_seleccionados)

    # Prediction button
    if st.button("Predecir Enfermedad"):
        # Create a dictionary for the input features
        # Initialize all symptoms to 0
        # Load the original dataframe to get the correct columns
        try:
            original_df = pd.read_excel("/content/Enfermedades y sintomas.xlsx")
            input_data = {col: 0 for col in original_df.columns if col != "Enfermedad_ES"}
        except FileNotFoundError:
            st.error("Error: Original data file not found. Cannot create input features.")
            st.stop()


        # Set selected symptoms to 1
        for s in sintomas_seleccionados:
            if s in input_data:
                input_data[s] = 1
            else:
                st.warning(f"⚠️ El síntoma '{s}' seleccionado no se encontró en los datos de entrenamiento.")


        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure the order of columns matches the training data
        # Load the original dataframe again to get the column order
        try:
            original_df = pd.read_excel("/content/Enfermedades y sintomas.xlsx")
            input_df = input_df[original_df.columns.drop("Enfermedad_ES")]
        except FileNotFoundError:
            st.error("Error: Original data file not found. Cannot match column order.")
            st.stop()


        # Predict the disease
        try:
            prediccion = modelo.predict(input_df)
            st.write("---")
            st.subheader("Resultado de la Predicción:")
            st.success(f"La enfermedad predicha es: **{prediccion[0]}**")
        except Exception as e:
            st.error(f"❌ Ocurrió un error durante la predicción: {e}")

