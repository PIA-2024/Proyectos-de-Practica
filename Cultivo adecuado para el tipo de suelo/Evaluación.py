# Importar librerías necesarias
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib  # Para cargar el modelo guardado
import seaborn as sns
import matplotlib.pyplot as plt

def run_evaluation():
    # Ruta del archivo preprocesado y el modelo entrenado
    data_file = 'C:/Users/piaal/Documents/PIA/DATA SCIENCE/PROYECTOS/Cultivo adecuado para el tipo de suelo/crop_data_preprocessed.csv'
    model_file = 'C:/Users/piaal/Documents/PIA/DATA SCIENCE/PROYECTOS/Cultivo adecuado para el tipo de suelo/crop_model.pkl'

    # Cargar el dataset preprocesado
    crop_data = pd.read_csv(data_file)

    # Dividir características (X) y la variable objetivo (y)
    X = crop_data.drop(columns=['label'])
    y = crop_data['label']

    # Dividir el dataset en conjuntos de entrenamiento y prueba
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Cargar el modelo entrenado
    model = joblib.load(model_file)
    print("Modelo cargado correctamente.")

    # Decodificar las clases numéricas a nombres originales
    class_names = [
        'apple', 'banana', 'blackgram', 'chickpea', 'coconut',
        'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans',
        'lentil', 'maize', 'mango', 'mothbeans', 'mungbean',
        'muskmelon', 'orange', 'papaya', 'pigeonpeas',
        'pomegranate', 'rice', 'watermelon'
    ]

    # Realizar predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    # 1. Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Verdadero")
    plt.savefig('confusion_matrix.png')  # Guardar la matriz como imagen
    plt.close()
    print("\nMatriz de Confusión guardada como 'confusion_matrix.png'.")

    # 2. Reporte de Clasificación
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("\nReporte de Clasificación:")
    print(report)

    # 3. Predicciones Personalizadas
    print("\nPredicción Personalizada:")
    sample = pd.DataFrame({
        'N': [50],
        'P': [40],
        'K': [20],
        'temperature': [25],
        'humidity': [80],
        'ph': [6.5],
        'rainfall': [200]
    })
    prediction = model.predict(sample)
    print(f"Valores de entrada: {sample.to_dict(orient='records')}")
    print(f"Predicción: {class_names[prediction[0]]}")

# Ejecutar solo si se llama directamente
if __name__ == "__main__":
    run_evaluation()
