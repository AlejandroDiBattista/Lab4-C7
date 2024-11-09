import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


## Crear Red Neuronal
class NeuralNetwork(nn.Module):
    def __init__(self, hidden_neurons):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1, hidden_neurons)  
        self.fc2 = nn.Linear(hidden_neurons, 1) 

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


## Leer Datos
data = pd.read_csv('ventas.csv')


## Normalizar Datos
X = data['dia'].values.reshape(-1, 1)
y = data['ventas'].values.reshape(-1, 1)

X_min = np.min(X)
X_max = np.max(X)
X_scaled = (X - X_min) / (X_max - X_min)

y_min = np.min(y)
y_max = np.max(y)
y_scaled = (y - y_min) / (y_max - y_min)


## Entrenar Red Neuronal
st.title("Clasificación de Ventas Diarias")
st.sidebar.header("Parámetros de la Red Neuronal")

learning_rate = st.sidebar.slider("Tasa de Aprendizaje", 0.0, 1.0, 0.1)
epochs = st.sidebar.slider("Cantidad de Épocas", 10, 10000, 100)
hidden_neurons = st.sidebar.slider("Neuronas en Capa Oculta", 1, 100, 5)

if st.sidebar.button("Entrenar"):
    # Dividir datos en entrenamiento y prueba
    train_size = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
    
    model = NeuralNetwork(hidden_neurons)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    progress_bar = st.sidebar.progress(0)
    loss_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(torch.FloatTensor(X_train))
        loss = criterion(outputs, torch.FloatTensor(y_train))
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        progress_bar.progress((epoch + 1) / epochs)

    st.sidebar.success("Entrenamiento finalizado con éxito!")


    ## Guardar Modelo
    torch.save(model.state_dict(), 'red_neuronal.pth')


    ## Graficar Predicciones
    plt.figure(figsize=(10,5))
    plt.plot(loss_history)
    plt.title('Evolución de la Función de Costo')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.grid()
    st.sidebar.pyplot(plt)

    model.eval()
    with torch.no_grad():
        predictions = model(torch.FloatTensor(X_scaled)).numpy()

    plt.figure(figsize=(10,5))
    plt.scatter(X, y, color='blue', label='Datos Reales')
    plt.plot(X, (predictions * (y_max - y_min)) + y_min, color='red', label='Predicciones')
    plt.title('Predicciones vs Ventas Reales')
    plt.xlabel('Día del Mes')
    plt.ylabel('Ventas')
    plt.legend()
    st.pyplot(plt)