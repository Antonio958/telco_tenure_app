# Telco Tenure App

Aplicación de ciencia de datos desarrollada en Python para el análisis y modelado de la duración de clientes (*tenure*) en una empresa de telecomunicaciones, utilizando técnicas de aprendizaje automático supervisado y no supervisado, bajo un enfoque modular y orientado a objetos.

---

## Objetivo del proyecto

Desarrollar una aplicación reproducible de ciencia de datos que permita analizar el comportamiento de la variable *tenure* y construir modelos predictivos de regresión, complementados con técnicas de clustering para identificar perfiles de clientes, integrando buenas prácticas de preprocesamiento, validación cruzada y evaluación de modelos.

---

## Estructura del proyecto

# Telco Tenure App

Aplicación de ciencia de datos desarrollada en Python para el análisis y modelado de la duración de clientes (*tenure*) en una empresa de telecomunicaciones, utilizando técnicas de aprendizaje automático supervisado y no supervisado, bajo un enfoque modular y orientado a objetos.

---

## Objetivo del proyecto

Desarrollar una aplicación reproducible de ciencia de datos que permita analizar el comportamiento de la variable *tenure* y construir modelos predictivos de regresión, complementados con técnicas de clustering para identificar perfiles de clientes, integrando buenas prácticas de preprocesamiento, validación cruzada y evaluación de modelos.

---

## Estructura del proyecto

elco_tenure_app/
│
├── main.py # Punto de entrada de la aplicación
├── importar_contenido_ejemplo.py # Ejemplo de carga y validación de datos
├── README.md
│
├── telco_app/ # Paquete principal de la aplicación
│ ├── init.py
│ ├── data_loader.py # Carga y validación del dataset
│ ├── preprocessing.py # Limpieza y transformación de datos
│ ├── eda.py # Análisis exploratorio
│ ├── modeling.py # Modelos de regresión y validación cruzada
│ └── clustering.py # Clustering no supervisado
│
├── exportaciones/
│ ├── figuras/ # Gráficas generadas por la aplicación
│ └── tablas/ # Resultados y métricas en formato CSV
│
└── biblioteca/
├── contexto_dataset.md
├── interpretacion_resultados.md
└── diagrama_clases.png # Diagrama de clases UML



---

## Dataset

El proyecto utiliza el conjunto de datos **Telco Customer Churn**, que contiene información demográfica, contractual y de servicios de clientes de telecomunicaciones.

El archivo CSV debe llamarse: WA_Fn-UseC_-Telco-Customer-Churn.csv




El dataset puede obtenerse desde Kaggle y debe colocarse en una ruta accesible para el script principal.  
En el caso de ejecución en Google Colab, el archivo puede cargarse desde Google Drive.

---

## Requisitos

- Python 3.9 o superior
- Bibliotecas principales:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

Las dependencias pueden instalarse con:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn


Ejecución de la aplicación
Ejecución local

Clonar el repositorio:

git clone https://github.com/Antonio958/telco_tenure_app.git
cd telco_tenure_app


Colocar el archivo CSV en la ruta configurada o ajustar la ruta en main.py.

Ejecutar la aplicación:

python main.py



Funcionamiento general

La aplicación ejecuta el siguiente flujo:

Carga y validación del dataset.

Análisis exploratorio de la variable tenure frente a distintas características.

Preprocesamiento de datos mediante imputación, escalado y codificación.

Entrenamiento de modelos de regresión (KNN como modelo principal).

Evaluación mediante validación cruzada y comparación con otros modelos.

Aplicación de clustering no supervisado para identificar perfiles de clientes.

Exportación de métricas, tablas y visualizaciones.

