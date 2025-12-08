import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


plt.style.use('default')
st.set_page_config(page_title="Estimación de Ventas Diarias", layout="wide")


st.sidebar.header("Parámetros de Entrenamiento")
learning_rate = st.sidebar.number_input('Aprendizaje', min_value=0.0001, max_value=1.0, value=0.01, step=0.0001, format="%.4f")
epochs = st.sidebar.number_input('Repeticiones', min_value=10, max_value=10000, value=100, step=10)
hidden_neurons = st.sidebar.number_input('Neuronas Capa Oculta', min_value=1, max_value=100, value=5)

np.random.seed(42)
dias = np.arange(1, 31)
ventas_base = 0.3 * (dias - 15) ** 2 + 80  
ruido = np.random.normal(0, 10, 30)  
ventas = ventas_base + ruido

data = pd.DataFrame({
    'dia': dias,
    'ventas': ventas
})


class RedNeuronal(nn.Module):
    def __init__(self, hidden_neurons):
        super(RedNeuronal, self).__init__()
        self.hidden1 = nn.Linear(1, hidden_neurons)
        self.hidden2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, 1)
    
    def forward(self, x):
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        x = self.output(x)
        return x


x_data = torch.tensor(dias.reshape(-1, 1), dtype=torch.float32) / 30 
y_data = torch.tensor(ventas.reshape(-1, 1), dtype=torch.float32)
scaler = MinMaxScaler()
y_data = torch.tensor(scaler.fit_transform(y_data), dtype=torch.float32)


if st.sidebar.button("Entrenar"):
    modelo = RedNeuronal(hidden_neurons)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(modelo.parameters(), lr=learning_rate)
    
 
    progress_text = f"Epoch {epochs}/{epochs} - Error: 0.0"
    progress_bar = st.progress(0, progress_text)
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = modelo(x_data)
        loss = criterion(outputs, y_data)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        if (epoch + 1) % 10 == 0:
            progress_text = f"Epoch {epoch + 1}/{epochs} - Error: {loss.item():.6f}"
            progress_bar.progress((epoch + 1) / epochs, progress_text)
    
    st.sidebar.success(f"Entrenamiento exitoso")
    

    fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
    ax_loss.plot(range(epochs), loss_history, 'g-', label='Pérdidas', linewidth=1)
    ax_loss.set_xlabel("Época")
    ax_loss.set_ylabel("Pérdida")
    ax_loss.grid(True, linestyle='--', alpha=0.7)
    st.sidebar.pyplot(fig_loss)
    

    with torch.no_grad():
        x_plot = torch.linspace(0, 1, 100).reshape(-1, 1)
        y_pred = modelo(x_plot)
        y_pred = scaler.inverse_transform(y_pred)
  
  
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(dias, ventas, color='blue', label='Datos Reales', s=30)
    ax.plot(x_plot * 30, y_pred, 'r-', label='Curva de Ajuste', linewidth=1.5)
    ax.set_xlabel("Día del Mes")
    ax.set_ylabel("Ventas")
    ax.set_title("Estimación de Ventas Diarias")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)

st.title("Estimación de Ventas Diarias")