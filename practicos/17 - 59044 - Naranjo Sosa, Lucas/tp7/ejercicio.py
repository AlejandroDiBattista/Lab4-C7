import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import streamlit as st

# Título de la aplicación Streamlit
st.title('Estimación de Ventas Diarias')

# Cargar datos desde un archivo CSV
df = pd.read_csv('ventas.csv')
dias = df['dia'].values
ventas = df['ventas'].values

# Normalizar los datos para el entrenamiento (escalar entre 0 y 1)
normalizador = MinMaxScaler()
dias_scaled = normalizador.fit_transform(dias.reshape(-1, 1))  # Normalización de los días
ventas_scaled = normalizador.fit_transform(ventas.reshape(-1, 1))  # Normalización de las ventas

# Sidebar para configurar los parámetros de entrenamiento
st.sidebar.header("Parámetros de Entrenamiento")
lr = st.sidebar.slider("Tasa de Aprendizaje", min_value=0.0, max_value=1.0, value=0.1)
epocas = st.sidebar.slider("Cantidad de Épocas", min_value=10, max_value=10000, value=100)
neuronas_ocultas = st.sidebar.slider("Neurona Capa Oculta", min_value=1, max_value=100, value=5)
entrenar_btn = st.sidebar.button("Entrenar")

# Definición de la red neuronal
class ModeloVentas(nn.Module):
    def __init__(self, neuronas_ocultas):
        super(ModeloVentas, self).__init__()
        self.capa_oculta = nn.Linear(1, neuronas_ocultas)  # Capa oculta
        self.capa_salida = nn.Linear(neuronas_ocultas, 1)  # Capa de salida

    def forward(self, x):
        x = torch.relu(self.capa_oculta(x))  # Función de activación ReLU
        x = self.capa_salida(x)
        return x

# Función para entrenar el modelo
def entrenar_modelo(lr, epocas, neuronas_ocultas):
    modelo = ModeloVentas(neuronas_ocultas)
    funcion_perdida = nn.MSELoss()  # Error cuadrático medio
    optimizador = optim.SGD(modelo.parameters(), lr=lr)  # Optimización por descenso de gradiente

    # Convertir los datos a tensores de PyTorch
    x_train = torch.tensor(dias_scaled, dtype=torch.float32)
    y_train = torch.tensor(ventas_scaled, dtype=torch.float32)

    errores = []

    # Mostrar progreso en Streamlit
    progreso_texto = st.empty()
    barra_progreso = st.progress(0)

    for epoca in range(epocas):
        optimizador.zero_grad()
        predicciones = modelo(x_train)
        error = funcion_perdida(predicciones, y_train)  # Calcular el error
        error.backward()  # Retropropagación del error
        optimizador.step()  # Actualización de los pesos

        errores.append(error.item())
        barra_progreso.progress((epoca + 1) / epocas)  # Actualizar la barra de progreso
        progreso_texto.text(f'Epoca {epoca + 1}/{epocas} - Error: {error.item():.6f}')

    return modelo, errores

# Entrenar el modelo si se presiona el botón
if entrenar_btn:
    modelo, errores = entrenar_modelo(lr, epocas, neuronas_ocultas)

    # Mostrar el gráfico de las pérdidas durante el entrenamiento
    fig, ax = plt.subplots()
    ax.plot(errores, color='green')
    ax.set_title("Pérdida durante el Entrenamiento")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida")
    st.pyplot(fig)

    # Realizar las predicciones con el modelo entrenado
    x_test = torch.tensor(dias_scaled, dtype=torch.float32)
    with torch.no_grad():  # Deshabilitar el cálculo de gradientes durante la predicción
        predicciones_scaled = modelo(x_test).numpy()

    # Desnormalizar las predicciones
    predicciones = normalizador.inverse_transform(predicciones_scaled)

    # Mostrar el gráfico con los datos reales y las predicciones
    fig, ax = plt.subplots()
    ax.scatter(dias, ventas, color='blue', label='Datos Reales')
    ax.plot(dias, predicciones, color='red', label='Predicciones')
    ax.set_title("Estimación de Ventas Diarias")
    ax.set_xlabel("Día del Mes")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)

    st.success("Entrenamiento completado con éxito")
