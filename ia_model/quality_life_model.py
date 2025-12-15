
#Importacion de librerias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("data/Mental_Health_and_Social_Media_Balance_Dataset.csv")

df.rename(columns={
    'Stress_Level(1-10)': 'Nivel de Estres',
    'Sleep_Quality(1-10)': 'Calidad del Sueño',
    'Daily_Screen_Time(hrs)': 'Tiempo en Pantalla',

}, inplace=True)

df['Estado de Salud'] = np.where(
    (df['Nivel de Estres'] > 6) |
    (df['Calidad del Sueño'] < 4)|
    (df['Tiempo en Pantalla'] > 6),
    0,
    1
)

X_training = df[['Nivel de Estres', 'Calidad del Sueño', 'Tiempo en Pantalla']]
y_training = df['Estado de Salud']

X_train, X_test, y_train, y_test = train_test_split(
    X_training, y_training, test_size=0.2, random_state=42, stratify=y_training
)

model = RandomForestClassifier(
    n_estimators = 100,
    max_depth = None,
    random_state = 42
)
model.fit(X_train, y_train)

joblib.dump(model, 'modelo_entrenado.pkl')