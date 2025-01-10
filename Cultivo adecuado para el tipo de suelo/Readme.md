# Predicción del Cultivo Adecuado en Función de Condiciones del Suelo y Clima 🌱

Este proyecto utiliza un modelo de Machine Learning para ayudar a agricultores a decidir qué cultivo sembrar basado en datos del suelo y el clima. Con este modelo, se pueden tomar decisiones informadas para optimizar el rendimiento de la tierra.

---

## **¿Qué Hace Este Proyecto?**
Imagina que un agricultor tiene una parcela de tierra y conoce las condiciones del suelo y el clima:
- **Nitrógeno (N), Fósforo (P), Potasio (K)**.
- **Temperatura, Humedad, pH y Precipitación**.

El modelo analiza estos datos y recomienda el cultivo más adecuado. Por ejemplo:

### Ejemplo de Uso
**Entrada:**\
- N: 50, P: 40, K: 20\
- Temperatura: 25°C, Humedad: 80%, pH: 6.5, Precipitación: 200 mm.

**Salida:**\
- **Recomendación:** Manzanas (apple).

### Ejemplo Real
El modelo puede analizar diferentes parcelas y sugerir:
- **Parcela A:** N: 60, P: 50, K: 40, Temperatura: 30°C, Humedad: 75%, pH: 6.8, Precipitación: 150 mm.\
  **Recomendación:** Maíz (maize).
- **Parcela B:** N: 30, P: 20, K: 10, Temperatura: 20°C, Humedad: 60%, pH: 5.5, Precipitación: 100 mm.\
  **Recomendación:** Lentejas (lentil).

---

## **Archivos Incluidos**
- **EDA.py:** Exploración inicial de los datos.
- **Procesamiento.py:** Preprocesamiento de datos (escalado y codificación).
- **Modelado.py:** Entrenamiento del modelo con Random Forest.
- **Evaluación.py:** Evaluación del modelo y predicciones personalizadas.
- **Informe_Final.ipynb:** Informe completo del proyecto en formato Jupyter Notebook.
- **Imágenes:** Gráficos generados durante el EDA (`mapa_calor_correlaciones.png`, etc.).

---

## **Resultados**
- **Exactitud del Modelo:** 100% en el conjunto de prueba.
- **Predicción Personalizada:** Entrada manual permite obtener recomendaciones inmediatas.

---

## **Recomendaciones Futuras**
1. Implementar validación cruzada para confirmar la robustez del modelo.
2. Explorar algoritmos adicionales como Gradient Boosting o redes neuronales.
3. Crear una aplicación web o móvil para integrar este modelo y hacerlo accesible a agricultores.
