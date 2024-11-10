import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import SGD
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

st.title('Estimación de Ventas Diarias')

@st.cache
def load_data():
    data = pd.read_csv('ventas.csv')
    return data

class VentasNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VentasNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

data = load_data()
scaler = MinMaxScaler()
data['ventas_norm'] = scaler.fit_transform(data[['ventas']])
x_data = torch.tensor(data['día'].values, dtype=torch.float32).view(-1, 1)
y_data = torch.tensor(data['ventas_norm'].values, dtype=torch.float32).view(-1, 1)

st.sidebar.header("Parámetros de la Red Neuronal")
learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.0, 1.0, 0.1, 0.01)
epochs = st.sidebar.slider("Cantidad de épocas", 10, 10000, 100, 10)
hidden_size = st.sidebar.slider("Neurona en la capa oculta", 1, 100, 5, 1)
train_button = st.sidebar.button("Entrenar")

if train_button:
    model = VentasNN(input_size=1, hidden_size=hidden_size, output_size=1)
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=learning_rate)
    progress_bar = st.progress(0)
    losses = []
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        output = model(x_data)
        loss = criterion(output, y_data)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        progress_bar.progress((epoch + 1) / epochs)
    st.success("Entrenamiento completado con éxito.")

    fig, ax = plt.subplots()
    ax.plot(range(epochs), losses, label="Función de costo")
    ax.set_xlabel("Épocas")
    ax.set_ylabel("Costo (MSE)")
    ax.set_title("Evolución de la Función de Costo")
    st.pyplot(fig)

    model.eval()
    with torch.no_grad():
        predictions = model(x_data).numpy()
        predictions = scaler.inverse_transform(predictions)

    fig, ax = plt.subplots()
    ax.plot(data['día'], data['ventas'], 'b-', label="Ventas Reales")
    ax.plot(data['día'], predictions, 'r--', label="Predicción de la Red Neuronal")
    ax.set_xlabel("Día del Mes")
    ax.set_ylabel("Ventas")
    ax.set_title("Ventas Reales vs. Predicción")
    ax.legend()
    st.pyplot(fig)
