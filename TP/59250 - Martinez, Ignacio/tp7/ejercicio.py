import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


## Crear Red Neuronal
## Leer Datos
## Normalizar Datos
## Entrenar Red Neuronal
## Guardar Modelo
## Graficar Predicciones

# Título 
st.title('Estimación de Ventas Diarias')

#  parámetros de la red 
# Tasa de aprendizaje
learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.0, 1.0, 0.1)
epochs = st.sidebar.slider("Cantidad de épocas", 10, 10000, 100)
hidden_neurons = st.sidebar.slider("Neurona en capa oculta", 1, 100, 5)

# Leer los datos 
data = pd.read_csv("ventas.csv")
# Extraer
X = data['dia'].values.reshape(-1, 1)
y = data['ventas'].values.reshape(-1, 1)

# normalización de datos 
1
X_norm = (X - X.min()) / (X.max() - X.min())
y_norm = (y - y.min()) / (y.max() - y.min())

# Definimos la red neuronal 
class VentasModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VentasModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Forward 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


X_tensor = torch.tensor(X_norm, dtype=torch.float32)
y_tensor = torch.tensor(y_norm, dtype=torch.float32)

# Función para entrenar la red neuronal 
def entrenar_red(model, X_tensor, y_tensor, epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    progreso = st.progress(0)

    # Ciclo de entrenamiento 
    for epoch in range(epochs):
        y_pred = model(X_tensor)
        # Calcula la pérdida 
        loss = criterion(y_pred, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progreso.progress((epoch + 1) / epochs)

    #  mensaje de éxito en la interfaz
    st.success("Entrenamiento completado")
    return model

#  guardar el modelo entrenado
def guardar_modelo(model, path="modelo_ventas.pth"):
    torch.save(model.state_dict(), path)
    st.write("Modelo guardado exitosamente en", path)

# Función para graficar 
def graficar_predicciones(model, X, y, X_norm, y_norm):
    
    with torch.no_grad():
        pred = model(torch.tensor(X_norm, dtype=torch.float32)).numpy()
    
    # Escala 
    pred_original = pred * (y.max() - y.min()) + y.min()

    
    plt.figure(figsize=(10, 5))
    plt.plot(X, y, label="Ventas reales", color="blue")
    plt.plot(X, pred_original, label="Predicción", color="red")
    plt.xlabel("Día")
    plt.ylabel("Ventas")
    plt.legend()
    
    st.pyplot(plt)

# Inicializamos la red neuronal 
model = VentasModel(1, hidden_neurons, 1)

# Botón 
if st.sidebar.button("Entrenar"):
    
    model = entrenar_red(model, X_tensor, y_tensor, epochs, learning_rate)
    
    graficar_predicciones(model, X, y, X_norm, y_norm)
    
    guardar_modelo(model)  
