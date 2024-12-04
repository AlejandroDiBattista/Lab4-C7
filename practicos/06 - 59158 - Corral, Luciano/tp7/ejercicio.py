import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


## Crear Red Neuronal
class VentasNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VentasNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.output(x)
        return x


data = pd.read_csv('ventas.csv')

dia = data['dia'].values.reshape(-1, 1)
ventas = data['ventas'].values.reshape(-1, 1)


dia_norm = dia / dia.max()
ventas_norm = ventas / ventas.max()

st.sidebar.title("Parámetros de la Red Neuronal")
learning_rate = st.sidebar.slider('Tasa de aprendizaje', 0.0, 1.0, 0.1)
epochs = st.sidebar.slider('Cantidad de épocas', 10, 10000, 100)
neurons = st.sidebar.slider('Cantidad de neuronas en la capa oculta', 1, 100, 5)

if st.sidebar.button("Entrenar"):
    model = VentasNN(input_size=1, hidden_size=neurons, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    X_train = torch.from_numpy(dia_norm).float()
    y_train = torch.from_numpy(ventas_norm).float()

    progress = st.progress(0)
    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress.progress((epoch + 1) / epochs)

    st.success("Entrenamiento finalizado con éxito!")
    plt.figure(figsize=(10, 5))
    plt.scatter(dia, ventas, label='Datos reales', color='blue')
    with torch.no_grad():
        predictions = model(X_train).numpy()
        plt.plot(dia, predictions * ventas.max(), label='Predicción', color='red')
    plt.xlabel('Día del Mes')
    plt.ylabel('Ventas')
    plt.legend()
    plt.title('Estimación de Ventas Diarias')
    st.pyplot(plt)
