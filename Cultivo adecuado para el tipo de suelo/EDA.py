# Configurar backend no interactivo para guardar gráficos
import matplotlib
matplotlib.use('Agg')

# Importar librerías
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda():
    # Ruta del archivo
    file_path = 'C:/Users/piaal/Documents/PIA/DATA SCIENCE/PROYECTOS/Cultivo adecuado para el tipo de suelo/Crop_recommendation.csv'

    # Cargar el dataset
    crop_data = pd.read_csv(file_path)

    # Mostrar información básica del dataset
    print("Información general del dataset:")
    print(crop_data.info())  # Información general
    print("\nPrimeras filas del dataset:")
    print(crop_data.head())  # Primeras filas
    print("\nEstadísticas básicas del dataset:")
    print(crop_data.describe())  # Estadísticas básicas

    # Visualizar distribuciones
    crop_data.hist(figsize=(12, 10), bins=20, color='skyblue', edgecolor='black')
    plt.suptitle("Distribuciones de las Variables", fontsize=16)
    plt.savefig('distribuciones_variables.png')  # Guardar el gráfico
    plt.close()

    # Relación entre variables: Mapa de calor de correlaciones
    plt.figure(figsize=(10, 8))
    sns.heatmap(crop_data.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Mapa de Calor de Correlaciones", fontsize=16)
    plt.savefig('mapa_calor_correlaciones.png')  # Guardar el gráfico
    plt.close()

    # Distribución de la variable objetivo (label)
    plt.figure(figsize=(12, 6))
    sns.countplot(data=crop_data, x='label', order=crop_data['label'].value_counts().index, palette="viridis")
    plt.xticks(rotation=90)
    plt.title("Distribución de Cultivos Recomendados", fontsize=16)
    plt.savefig('distribucion_cultivos.png')  # Guardar el gráfico
    plt.close()

    # Guardar los datos para el siguiente paso (opcional)
    output_file = 'C:/Users/piaal/Documents/PIA/DATA SCIENCE/PROYECTOS/Cultivo adecuado para el tipo de suelo/crop_data_cleaned.csv'
    crop_data.to_csv(output_file, index=False)
    print(f"\nDatos limpios guardados en {output_file}")

# Ejecutar solo si se llama directamente
if __name__ == "__main__":
    run_eda()
