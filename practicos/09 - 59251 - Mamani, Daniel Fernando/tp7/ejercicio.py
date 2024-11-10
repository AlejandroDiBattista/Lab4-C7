import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Título de la aplicación
st.title('Estimación de Ventas Diarias')

# Parámetros de la red neuronal
learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.0, 1.0, 0.1)
epochs = st.sidebar.slider("Cantidad de épocas", 10, 10000, 100)
hidden_neurons = st.sidebar.slider("Neuronas en capa oculta", 1, 100, 5)

# Cargar datos desde ventas.csv
data = pd.read_csv("C:/Users/FERNANDO/Documents/GitHub/Lab4-C7/Lab4-C7/practicos/09 - 59251 - Mamani, Daniel Fernando/tp7/ventas.csv")
st.write("Datos de ventas:", data)

# Normalizar los datos
X = np.array(data['dia']).reshape(-1, 1).astype(np.float32)  # Cambié 'día' a 'dia'
y = np.array(data['ventas']).reshape(-1, 1).astype(np.float32)
X = (X - X.mean()) / X.std()  # Normalización de X

# Definir la red neuronal
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

# Inicializar la red y el optimizador
model = SimpleNN(input_size=1, hidden_size=hidden_neurons, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Botón para iniciar el entrenamiento
if st.button("Entrenar"):
    # Entrenamiento de la red neuronal
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    progress_bar = st.progress(0)
    loss_values = []
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)

        # Backward pass y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Guardar la pérdida y actualizar barra de progreso
        loss_values.append(loss.item())
        progress_bar.progress((epoch + 1) / epochs)
    
    st.success("Entrenamiento finalizado con éxito")

    # Gráfico de la función de costo
    st.subheader("Evolución de la función de costo")
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values)
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.title("Evolución de la función de costo")
    st.pyplot(plt)

    # Predicciones y gráfico de ventas
    st.subheader("Predicción de ventas diarias")
    with torch.no_grad():
        predictions = model(X_tensor).numpy()

    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, label="Datos reales")
    plt.plot(X, predictions, color="red", label="Predicción de la red neuronal")
    plt.xlabel("Día (normalizado)")
    plt.ylabel("Ventas")
    plt.legend()
    plt.title("Ventas vs Predicción")
    st.pyplot(plt)
