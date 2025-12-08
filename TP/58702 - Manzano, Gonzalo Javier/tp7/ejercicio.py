import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Configuración de la página
st.title('Estimación de Ventas Diarias')

# Cargar datos
ventas_df = pd.read_csv('ventas.csv')
dias = ventas_df['dia'].values
ventas = ventas_df['ventas'].values

# Normalización de los datos
scaler = MinMaxScaler()
dias_normalizados = scaler.fit_transform(dias.reshape(-1, 1))
ventas_normalizadas = scaler.fit_transform(ventas.reshape(-1, 1))

# Parámetros de entrada
st.sidebar.header("Parámetros de Entrenamiento")
learning_rate = st.sidebar.slider("Tasa de Aprendizaje", 0.0, 1.0, 0.1)
epochs = st.sidebar.slider("Cantidad de Épocas", 10, 10000, 100)
hidden_neurons = st.sidebar.slider("Neurona Capa Oculta", 1, 100, 5)
boton_entrenar = st.sidebar.button("Entrenar")

# Clase de la red neuronal
class RedNeuronal(nn.Module):
    def __init__(self, hidden_neurons):
        super(RedNeuronal, self).__init__()
        self.hidden = nn.Linear(1, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, 1)
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

# Función para entrenar la red
def entrenar_red(learning_rate, epochs, hidden_neurons):
    red = RedNeuronal(hidden_neurons)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(red.parameters(), lr=learning_rate)

    x_train = torch.tensor(dias_normalizados, dtype=torch.float32)
    y_train = torch.tensor(ventas_normalizadas, dtype=torch.float32)

    losses = []

    progreso = st.empty()
    barra = st.progress(0)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = red(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        barra.progress((epoch + 1) / epochs)
        progreso.text(f'Epoch {epoch + 1}/{epochs} - Error: {loss.item():.6f}')

    return red, losses

# Entrenar y graficar
if boton_entrenar:
    red, losses = entrenar_red(learning_rate, epochs, hidden_neurons)

    # Gráfico de pérdida
    fig, ax = plt.subplots()
    ax.plot(losses, color='green')
    ax.set_title("Pérdida durante el Entrenamiento")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida")
    st.pyplot(fig)

    # Predicciones
    x_test = torch.tensor(dias_normalizados, dtype=torch.float32)
    with torch.no_grad():
        predicciones_normalizadas = red(x_test).numpy()
    predicciones = scaler.inverse_transform(predicciones_normalizadas)

    # Gráfico de Ventas
    fig, ax = plt.subplots()
    ax.scatter(dias, ventas, color='blue', label='Datos Reales')
    ax.plot(dias, predicciones, color='red', label='Curva de Ajuste')
    ax.set_title("Estimación de Ventas Diarias")
    ax.set_xlabel("Día del Mes")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)

    st.success("Entrenamiento exitoso")
