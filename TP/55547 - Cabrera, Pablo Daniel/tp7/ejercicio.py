import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def configurar_pagina():
    st.title("Estimación de Ventas Diarias")
    return {
        "learning_rate": st.sidebar.number_input("Aprendizaje", min_value=0.0001, max_value=1.0, value=0.01, step=0.0001, format="%.4f"),
        "epochs": st.sidebar.number_input("Repeticiones", min_value=10, max_value=10000, value=1000, step=10),
        "hidden_neurons": st.sidebar.number_input("Neuronas Capa Oculta", min_value=1, max_value=100, value=10)
    }

def generar_datos():
    np.random.seed(42)
    dias = np.arange(1, 31)
    ventas_base = 0.3 * (dias - 15) ** 2 + 80
    ruido = np.random.normal(0, 10, 30)
    ventas = ventas_base + ruido
    return dias, ventas

class RedNeuronal(nn.Module):
    def __init__(self, hidden_neurons):
        super().__init__()
        self.hidden1 = nn.Linear(1, hidden_neurons)
        self.hidden2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, 1)
        
    def forward(self, x):
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        return self.output(x)

def preparar_datos(dias, ventas):
    x_data = torch.tensor(dias.reshape(-1, 1), dtype=torch.float32) / 30
    y_data = torch.tensor(ventas.reshape(-1, 1), dtype=torch.float32)
    scaler = MinMaxScaler()
    y_data = torch.tensor(scaler.fit_transform(y_data), dtype=torch.float32)
    return x_data, y_data, scaler

def entrenar_modelo(modelo, x_data, y_data, params):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(modelo.parameters(), lr=params["learning_rate"])
    epoch_text = st.sidebar.empty()
    loss_history = []

    for epoch in range(params["epochs"]):
        optimizer.zero_grad()
        outputs = modelo(x_data)
        loss = criterion(outputs, y_data)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (epoch + 1) % 10 == 0 or epoch == params["epochs"] - 1:
            epoch_text.text(f"Época {epoch + 1}/{params['epochs']} - Error: {loss.item():.6f}")
            
    return loss_history

def visualizar_perdidas(loss_history, epochs):
    fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
    ax_loss.plot(range(epochs), loss_history, "g-", label="Pérdidas", linewidth=1)
    ax_loss.set_xlabel("Época")
    ax_loss.set_ylabel("Pérdida")
    ax_loss.grid(True, linestyle="--", alpha=0.7)
    st.sidebar.pyplot(fig_loss)

def visualizar_predicciones(modelo, dias, ventas, scaler):
    with torch.no_grad():
        x_plot = torch.linspace(0, 1, 100).reshape(-1, 1)
        y_pred = modelo(x_plot)
        y_pred = scaler.inverse_transform(y_pred)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(dias, ventas, color="blue", label="Datos Reales", s=30)
    ax.plot(x_plot * 30, y_pred, "r-", label="Curva de Ajuste", linewidth=1.5)
    ax.set_xlabel("Día del Mes")
    ax.set_ylabel("Ventas")
    ax.set_title("Estimación de Ventas Diarias")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig)

def main():
    params = configurar_pagina()
    dias, ventas = generar_datos()
    x_data, y_data, scaler = preparar_datos(dias, ventas)
    
    if st.sidebar.button("Entrenar"):
        modelo = RedNeuronal(params["hidden_neurons"])
        loss_history = entrenar_modelo(modelo, x_data, y_data, params)
        
        st.sidebar.success("Entrenamiento completado")
        visualizar_perdidas(loss_history, params["epochs"])
        visualizar_predicciones(modelo, dias, ventas, scaler)

if __name__ == "__main__":
    main()