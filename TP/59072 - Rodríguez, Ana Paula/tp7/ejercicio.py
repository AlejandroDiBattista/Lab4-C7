import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Configuración de Streamlit
st.title('Estimación de Ventas Diarias')
st.sidebar.title("Parámetros de Entrenamiento")

# Cargar y visualizar datos
# Asume que tienes un archivo 'ventas.csv' en el mismo directorio
# con columnas 'día' (int) y 'ventas' (float)
data = pd.read_csv(r'C:/Users/pauro/OneDrive/Documentos/GitHub/Lab4-C7/practicos/28 - 59072 - Rodríguez, Ana Paula/tp7/ventas.csv')
st.write("Visualización de datos de ventas diarias")
fig, ax = plt.subplots()
ax.plot(data['día'], data['ventas'], label="Ventas Reales")
ax.set_xlabel("día")
ax.set_ylabel("Ventas")
st.pyplot(fig)

# Parámetros de la red neuronal configurados desde el sidebar
learning_rate = st.sidebar.slider("Tasa de Aprendizaje", 0.0, 1.0, 0.1)
epochs = st.sidebar.slider("Cantidad de Épocas", 10, 10000, 100)
hidden_neurons = st.sidebar.slider("Neuronas en la Capa Oculta", 1, 100, 5)

# Definir la red neuronal
class VentasNet(nn.Module):
    def __init__(self, hidden_neurons):
        super(VentasNet, self).__init__()
        self.hidden = nn.Linear(1, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, 1)
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

# Función para normalizar los datos
def normalize_data(data):
    return (data - data.min()) / (data.max() - data.min())

# Normalizar los datos de entrada y salida
data['día_norm'] = normalize_data(data['día'])
data['ventas_norm'] = normalize_data(data['ventas'])

# Botón para iniciar el entrenamiento
if st.sidebar.button("Entrenar"):
    # Convertir los datos normalizados a tensores
    x_data = torch.tensor(data['día_norm'].values, dtype=torch.float32).view(-1, 1)
    y_data = torch.tensor(data['ventas_norm'].values, dtype=torch.float32).view(-1, 1)

    # Inicializar la red, la función de pérdida y el optimizador
    net = VentasNet(hidden_neurons)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    # Barra de progreso
    progress_bar = st.progress(0)
    losses = []

    # Entrenamiento de la red
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = net(x_data)
        loss = criterion(outputs, y_data)
        loss.backward()
        optimizer.step()
        
        # Guardar la pérdida en cada época
        losses.append(loss.item())

        # Actualizar la barra de progreso
        progress_bar.progress((epoch + 1) / epochs)

    # Guardar el modelo entrenado (opcional)
    torch.save(net.state_dict(), 'modelo_ventas.pth')
    st.sidebar.success("Entrenamiento completado con éxito")

    # Graficar la función de costo
    fig, ax = plt.subplots()
    ax.plot(range(epochs), losses, label="Pérdida")
    ax.set_xlabel("Épocas")
    ax.set_ylabel("Costo")
    st.write("Evolución de la función de costo")
    st.pyplot(fig)

    # Predicciones finales y graficar resultados
    with torch.no_grad():
        predictions = net(x_data).numpy() * (data['ventas'].max() - data['ventas'].min()) + data['ventas'].min()

    fig, ax = plt.subplots()
    ax.plot(data['día'], data['ventas'], label="Ventas Reales")
    ax.plot(data['día'], predictions, label="Predicción", linestyle='--')
    ax.set_xlabel("día")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.write("Datos de ventas con predicción de la red neuronal")
    st.pyplot(fig)
