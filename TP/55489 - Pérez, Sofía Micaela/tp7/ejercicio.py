import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Título de la aplicación
st.title('Estimación de Ventas Diarias con Red Neuronal')

# Cargar el dataset
ventas_df = pd.read_csv('ventas.csv')
x_data = torch.tensor(ventas_df['dia'].values, dtype=torch.float32).reshape(-1, 1)
y_data = torch.tensor(ventas_df['ventas'].values, dtype=torch.float32).reshape(-1, 1)

# Mostrar el gráfico de ventas
st.subheader("Datos de Ventas")
fig, ax = plt.subplots()
ax.plot(ventas_df['dia'], ventas_df['ventas'], label="Ventas", color='blue')
ax.set_xlabel("Día")
ax.set_ylabel("Ventas")
ax.legend()
st.pyplot(fig)

# Panel izquierdo para ingresar parámetros
st.sidebar.header("Parámetros de la Red Neuronal")
learning_rate = st.sidebar.slider("Tasa de Aprendizaje", 0.0, 1.0, 0.1, 0.01)
epochs = st.sidebar.slider("Cantidad de Épocas", 10, 10000, 100)
hidden_neurons = st.sidebar.slider("Neurona en Capa Oculta", 1, 100, 5)
train_button = st.sidebar.button("Entrenar")

# Definición de la red neuronal
class SimpleNN(nn.Module):
    def __init__(self, hidden_neurons):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(1, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

# Entrenamiento de la red
if train_button:
    # Instanciar la red y definir función de pérdida y optimizador
    model = SimpleNN(hidden_neurons)
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=learning_rate)
    
    # Mostrar barra de progreso
    progress_bar = st.progress(0)
    loss_list = []
    
    for epoch in tqdm(range(epochs)):
        # Forward pass
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        loss_list.append(loss.item())
        
        # Backward pass y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Actualizar barra de progreso
        if epoch % (epochs // 100) == 0:
            progress_bar.progress(epoch / epochs)
    
    progress_bar.progress(1.0)  # Completar la barra al final
    st.success("Entrenamiento finalizado con éxito")
    
    # Gráfico de la función de pérdida
    st.subheader("Evolución de la Función de Costo")
    fig, ax = plt.subplots()
    ax.plot(loss_list, color='red')
    ax.set_xlabel("Épocas")
    ax.set_ylabel("Pérdida")
    st.pyplot(fig)
    
    # Predicciones y gráfico de comparación
    with torch.no_grad():
        y_pred = model(x_data)
    
    st.subheader("Ventas vs. Predicción")
    fig, ax = plt.subplots()
    ax.plot(ventas_df['dia'], ventas_df['ventas'], label="Ventas Reales", color='blue')
    ax.plot(ventas_df['dia'], y_pred.numpy(), label="Predicción", color='orange')
    ax.set_xlabel("Día")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)


## Crear Red Neuronal
## Leer Datos
## Normalizar Datos
## Entrenar Red Neuronal
## Guardar Modelo
## Graficar Predicciones
st.title('Estimación de Ventas Diarias')