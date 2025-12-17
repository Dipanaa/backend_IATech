from flask import Flask, jsonify, request
import joblib
import numpy as np
import os
from dotenv import load_dotenv
import google.generativeai as genai
from flask_cors import CORS

#Creacion de la instancia
app = Flask(__name__)
CORS(app)

#Cargar entorno gemini
load_dotenv(dotenv_path="environment/developer.env")

#gemini key
gemini_api_key = os.getenv("GEMINI_API_KEY")

if gemini_api_key is None:
    print("Clave no encontrada")


#Carga de modelo IA
try:
    # Rutra de modelo pkl en binario
    RUTA_MODELO = 'modelo_entrenado.pkl' 
    modelo_ia = joblib.load(RUTA_MODELO)
    print("Modelo de IA cargado con éxito en memoria.")
except Exception as e:
    print(f"ERROR: No se pudo cargar el modelo. ¿Está en la ruta correcta? {e}")
    modelo_ia = None

#Endpoints
#POST predict endpoint 
@app.route("/predict",methods=["POST"])
async def predict():
    if modelo_ia is None:
        return jsonify({"error": "Modelo no disponible, el servidor falló al cargarlo."}), 503

    # Datos recibidos por json
    data = request.json
    
    # Validación y Extracción de datos
    try:
        input_data = np.array([
            data['Nivel_Estres'], 
            data['Calidad_Sueno'], 
            data['Tiempo_Pantalla']
        ]).reshape(1, -1) # Formato para construccion de array 2D con filas y columnas
    except KeyError:
        return jsonify({"error": "Faltan datos de entrada ('Nivel_Estres', 'Calidad_Sueno', 'Tiempo_Pantalla')."}), 400

    # 2. Predicción
    prediccion = modelo_ia.predict(input_data)[0]
    
    # 3. Formateo de la respuesta
    resultado = "Saludable" if prediccion == 1 else "No Saludable"

    #Entregar resultados con gemini

    model = genai.GenerativeModel("gemini-2.5-flash")

    #Prompt de analisis
    prompt = f"""
    Tu tarea como modelo es entregar recomendaciones de salud digital que deben ser solo 3 en base a
    la respuesta que entrege un modelo predictivo. Esta puede ser Saludable o No Saludable.
    Solo debes entregar recomendaciones si la respuesta es No Saludable si no debes decir lo siguiente:
    Usted presenta buenos habitos de salud digital, por favor continue de esa forma.

    RESPUESTA : {resultado}

    No debes entregar informacion fuera del contexto de salud digital
    """

    respuesta = ""

    try:
        response = await model.generate_content_async(prompt)
        respuesta = response.text
    except:
        respuesta = "Servicio de recomendacion de IA no disponible"



    return jsonify({
        "prediccion_numerica": int(prediccion),
        "estado_salud": resultado,
        "recomendacion_ia": respuesta
    })


if __name__ == '__main__':
    app.run(debug=True)