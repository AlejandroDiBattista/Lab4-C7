import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Usar un estilo predefinido de Matplotlib
plt.style.use('ggplot')  # Cambié el estilo a ggplot
st.set_page_config(page_title="Pronóstico de Ventas Diarias", layout="wide")

st.sidebar.header("Parámetros de Modelado")
learning_rate = st.sidebar.number_input('Tasa de Aprendizaje', min_value=0.0001, max_value=1.0, value=0.02, step=0.0001, format="%.4f")
epochs = st.sidebar.number_input('Número de Épocas', min_value=10, max_value=10000, value=200, step=10)
hidden_neurons = st.sidebar.number_input('Neuronas en Capa Oculta', min_value=1, max_value=150, value=10)

np.random.seed(100)
days = np.arange(1, 31)
base_sales = 0.25 * (days - 10) ** 2 + 75  
noise = np.random.normal(0, 12, 30)  
sales = base_sales + noise

data = pd.DataFrame({
    'día': days,
    'ventas': sales
})

class ModeloNN(nn.Module):
    def __init__(self, hidden_neurons):
        super(ModeloNN, self).__init__()
        self.capa_oculta1 = nn.Linear(1, hidden_neurons)
        self.capa_oculta2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.salida = nn.Linear(hidden_neurons, 1)
    
    def forward(self, x):
        x = torch.tanh(self.capa_oculta1(x))
        x = torch.tanh(self.capa_oculta2(x))
        x = self.salida(x)
        return x

x_datos = torch.tensor(days.reshape(-1, 1), dtype=torch.float32) / 30 
y_datos = torch.tensor(sales.reshape(-1, 1), dtype=torch.float32)
scaler = MinMaxScaler()
y_datos = torch.tensor(scaler.fit_transform(y_datos), dtype=torch.float32)

if st.sidebar.button("Iniciar Entrenamiento"):
    modelo = ModeloNN(hidden_neurons)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(modelo.parameters(), lr=learning_rate)
    
    progreso_texto = f"Época {epochs}/{epochs} - Error: 0.0"
    progreso_barra = st.progress(0, progreso_texto)
    historial_perdidas = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        salidas = modelo(x_datos)
        perdida = criterion(salidas, y_datos)
        perdida.backward()
        optimizer.step()
        
        historial_perdidas.append(perdida.item())
        if (epoch + 1) % 10 == 0:
            progreso_texto = f"Época {epoch + 1}/{epochs} - Error: {perdida.item():.6f}"
            progreso_barra.progress((epoch + 1) / epochs, progreso_texto)
    
    st.sidebar.success(f"Entrenamiento completado con éxito")
    
    fig_perdidas, ax_perdidas = plt.subplots(figsize=(6, 4))
    ax_perdidas.plot(range(epochs), historial_perdidas, 'b-', label='Perdida', linewidth=1)
    ax_perdidas.set_xlabel("Época")
    ax_perdidas.set_ylabel("Perdida")
    ax_perdidas.grid(True, linestyle='--', alpha=0.7)
    st.sidebar.pyplot(fig_perdidas)
    
    with torch.no_grad():
        x_plot = torch.linspace(0, 1, 100).reshape(-1, 1)
        y_pred = modelo(x_plot)
        y_pred = scaler.inverse_transform(y_pred)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(days, sales, color='green', label='Datos Reales', s=30)
    ax.plot(x_plot * 30, y_pred, 'r-', label='Predicción Ajustada', linewidth=1.5)
    ax.set_xlabel("Día del Mes")
    ax.set_ylabel("Ventas")
    ax.set_title("Pronóstico de Ventas Diarias")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)

st.title("Pronóstico de Ventas Diarias")
