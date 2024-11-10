import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=10, output_dim=1):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu_activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu_activation(x)
        x = self.layer2(x)
        return x

# Cargar los datos de ventas
data = pd.read_csv('ventas.csv')
days = data['dia'].values
sales = data['ventas'].values

# Función para normalizar datos
def scale_data(input_data):
    mean = input_data.mean()
    std_dev = input_data.std()
    return (input_data - mean) / std_dev, mean, std_dev

# Configuración de la interfaz con Streamlit
st.sidebar.header('Configuración de Entrenamiento')

# Parámetros de entrada de aprendizaje
learning_rate_input = st.sidebar.number_input('Tasa de Aprendizaje', min_value=0.0001, max_value=1.0, value=0.01, format='%0.4f')
num_epochs_input = st.sidebar.number_input('Número de Épocas', min_value=10, max_value=10000, value=100, step=10)
hidden_layer_neurons = st.sidebar.number_input('Neurona en Capa Oculta', min_value=1, max_value=100, value=10)

# Normalizar los datos
days_scaled, days_mean, days_std = scale_data(days)
sales_scaled, sales_mean, sales_std = scale_data(sales)

# Convertir los datos normalizados en tensores
days_tensor = torch.FloatTensor(days_scaled.reshape(-1, 1))
sales_tensor = torch.FloatTensor(sales_scaled.reshape(-1, 1))

# Inicializar el modelo y los optimizadores
model = NeuralNetwork(hidden_dim=hidden_layer_neurons)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_input)

if st.sidebar.button('Entrenar Modelo'):
    progress_bar = st.progress(0)
    progress_text = st.empty()
    loss_history = []

    for epoch in range(num_epochs_input):
        # Predicción del modelo
        predictions = model(days_tensor)
        loss = loss_function(predictions, sales_tensor)

        # Actualizar gradientes
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        # Actualizar barra de progreso cada 10 épocas
        if (epoch + 1) % 10 == 0:
            progress_bar.progress((epoch + 1) / num_epochs_input)
            progress_text.text(f'Epoca {epoch+1}/{num_epochs_input} - Pérdida: {loss.item():.6f}')
    
    st.sidebar.success('Entrenamiento Completo')

    # Mostrar gráfico de las pérdidas
    fig_loss, ax_loss = plt.subplots(figsize=(6, 2))
    ax_loss.plot(loss_history, color='orange', label='Pérdida durante el Entrenamiento')
    ax_loss.set_xlabel('Épocas')
    ax_loss.set_ylabel('Pérdida')
    ax_loss.grid(True)
    st.sidebar.pyplot(fig_loss)

# Realizar predicción sobre nuevos datos
with torch.no_grad():
    test_days = torch.FloatTensor(np.linspace(days_scaled.min(), days_scaled.max(), 100).reshape(-1, 1))
    predicted_sales_scaled = model(test_days)

    # Desnormalizar predicciones
    test_days_denorm = test_days.numpy() * days_std + days_mean
    predicted_sales_denorm = predicted_sales_scaled.numpy() * sales_std + sales_mean

# Graficar los resultados
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(days, sales, color='blue', label='Datos Reales')
ax.plot(test_days_denorm, predicted_sales_denorm, color='red', label='Predicciones del Modelo', linewidth=2)
ax.set_xlabel('Día del Mes')
ax.set_ylabel('Ventas')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
ax.grid(True)
st.pyplot(fig)
