# Importar librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # Para guardar el modelo entrenado

def run_model_training():
    # Ruta del archivo preprocesado
    file_path = 'C:/Users/piaal/Documents/PIA/DATA SCIENCE/PROYECTOS/Cultivo adecuado para el tipo de suelo/crop_data_preprocessed.csv'

    # Cargar el dataset preprocesado
    crop_data = pd.read_csv(file_path)

    # Dividir características (X) y la variable objetivo (y)
    X = crop_data.drop(columns=['label'])
    y = crop_data['label']

    # Dividir el dataset en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")

    # Entrenar un modelo de clasificación (Random Forest)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluar el modelo en el conjunto de prueba
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Exactitud del modelo: {accuracy:.2f}")

    # Guardar el modelo entrenado
    model_file = 'C:/Users/piaal/Documents/PIA/DATA SCIENCE/PROYECTOS/Cultivo adecuado para el tipo de suelo/crop_model.pkl'
    joblib.dump(model, model_file)
    print(f"Modelo guardado en: {model_file}")

# Ejecutar solo si se llama directamente
if __name__ == "__main__":
    run_model_training()
