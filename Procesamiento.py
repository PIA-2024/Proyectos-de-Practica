# Importar librerías necesarias
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def run_preprocessing():
    # Ruta del archivo limpio generado en el EDA
    file_path = 'C:/Users/piaal/Documents/PIA/DATA SCIENCE/PROYECTOS/Cultivo adecuado para el tipo de suelo/crop_data_cleaned.csv'

    # Cargar el dataset limpio
    crop_data = pd.read_csv(file_path)

    print("Primeras filas del dataset limpio cargado:")
    print(crop_data.head())

    # 1. Normalizar o Escalar Variables Numéricas
    scaler = MinMaxScaler()  # Crear el objeto escalador
    numerical_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    crop_data[numerical_columns] = scaler.fit_transform(crop_data[numerical_columns])

    print("\nDatos normalizados (primeras filas):")
    print(crop_data[numerical_columns].head())

    # 2. Codificar la Variable Categórica 'label'
    label_encoder = LabelEncoder()  # Crear el objeto codificador
    crop_data['label'] = label_encoder.fit_transform(crop_data['label'])

    print("\nClases codificadas:")
    print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

    print("\nDatos preprocesados (primeras filas):")
    print(crop_data.head())

    # Guardar los datos preprocesados para el siguiente paso
    output_file = 'C:/Users/piaal/Documents/PIA/DATA SCIENCE/PROYECTOS/Cultivo adecuado para el tipo de suelo/crop_data_preprocessed.csv'
    crop_data.to_csv(output_file, index=False)
    print(f"\nDatos preprocesados guardados en {output_file}")

# Ejecutar solo si se llama directamente
if __name__ == "__main__":
    run_preprocessing()
