# Regresión Lineal Simple

#Explicaciones clave:
#Importación de datos: Se asume que el archivo Salary_Data.csv contiene dos columnas: años de experiencia y salario.
#División del dataset: Usamos train_test_split para dividir los datos en un conjunto de entrenamiento (2/3) y prueba (1/3).
#Entrenamiento del modelo: Utilizamos LinearRegression de sklearn para ajustar un modelo lineal a los datos de entrenamiento.
#Visualización de resultados:
#Gráfico para el conjunto de entrenamiento: muestra los puntos reales y la línea de regresión ajustada.
#Gráfico para el conjunto de prueba: muestra los puntos reales del conjunto de prueba y la misma línea ajustada con el conjunto de entrenamiento.

# Importar las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset de Netflix
dataset = pd.read_csv('netflix_titles.csv')

# Preprocesamiento de datos:
# Relacionaremos el año de lanzamiento ('release_year') con la duración de las películas ('duration')

# 1. Filtrar solo películas
movies = dataset[dataset['type'] == 'Movie'].copy()

# 2. Convertir 'duration' a minutos numéricos (eliminar ' min' y convertir a float)
movies['duration_min'] = movies['duration'].str.replace(' min', '').astype(float)

# 3. Seleccionar solo las columnas que necesitamos y eliminar filas con valores nulos
netflix_data = movies[['release_year', 'duration_min']].dropna()

# Definir variables X (año de lanzamiento) e y (duración en minutos)
X = netflix_data.iloc[:, :-1].values  # Variable independiente (año de lanzamiento)
y = netflix_data.iloc[:, -1].values   # Variable dependiente (duración en minutos)

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Entrenar el modelo de Regresión Lineal Simple
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predecir los resultados del conjunto de prueba
y_pred = regressor.predict(X_test)

# Visualizar los resultados del conjunto de entrenamiento
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Duración vs Año de Lanzamiento (Conjunto de Entrenamiento)')
plt.xlabel('Año de Lanzamiento')
plt.ylabel('Duración de la Película (minutos)')
plt.show()

# Visualizar los resultados del conjunto de prueba
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Duración vs Año de Lanzamiento (Conjunto de Prueba)')
plt.xlabel('Año de Lanzamiento')
plt.ylabel('Duración de la Película (minutos)')
plt.show()

#Conclusion
#El modelo de regresión lineal simple aplicado al dataset de Netflix permitió analizar si existe una relación entre el año de lanzamiento y la duración de las películas. 
#Los resultados muestran que esta relación es muy débil, lo que indica que el año no influye significativamente en la duración. 
#Este análisis ayuda a entender tendencias generales y demuestra cómo aplicar técnicas de regresión a datos reales.
