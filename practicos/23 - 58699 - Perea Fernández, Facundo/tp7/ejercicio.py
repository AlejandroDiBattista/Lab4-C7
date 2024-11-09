import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


## Crear Red Neuronal
class RedNeuronal(nn.Module):
    def __init__(self, neuronas_ocultas):
        super(RedNeuronal, self).__init__()
        self.fc1 = nn.Linear(1, neuronas_ocultas)  
        self.fc2 = nn.Linear(neuronas_ocultas, 1) 

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


## Inicializar session_state
if 'tasa_aprendizaje' not in st.session_state:
    st.session_state.tasa_aprendizaje = 0.1

if 'epocas' not in st.session_state:
    st.session_state.epocas = 100

if 'neuronas_ocultas' not in st.session_state:
    st.session_state.neuronas_ocultas = 5

if 'modelo_entrenado' not in st.session_state:
    st.session_state.modelo_entrenado = False


## Entrenar Red Neuronal
st.title("Predicción de Ventas Diarias")
st.sidebar.header("Parámetros de la Red Neuronal")

st.session_state.tasa_aprendizaje = st.sidebar.slider("Tasa de Aprendizaje", 0.0, 1.0, st.session_state.tasa_aprendizaje)
st.session_state.epocas = st.sidebar.slider("Cantidad de Épocas", 10, 10000, st.session_state.epocas)
st.session_state.neuronas_ocultas = st.sidebar.slider("Neuronas en Capa Oculta", 1, 100, st.session_state.neuronas_ocultas)

if st.sidebar.button("Entrenar"):
    model = RedNeuronal(st.session_state.neuronas_ocultas)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=st.session_state.tasa_aprendizaje)

    progress_bar = st.sidebar.progress(0)
    loss_history = []

    for epoch in range(st.session_state.epocas):
        optimizer.zero_grad()
        outputs = model(torch.FloatTensor(X_scaled))
        loss = criterion(outputs, torch.FloatTensor(y_scaled))
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
        progress_bar.progress((epoch + 1) / st.session_state.epocas)

    st.success("Entrenamiento finalizado con éxito")
    
    ## Guardar Modelo
    torch.save(model.state_dict(), 'red_neuronal.pth')
    
    st.session_state.modelo_entrenado = True


## Graficar Predicciones
if st.session_state.modelo_entrenado:
    model = RedNeuronal(st.session_state.neuronas_ocultas)
    model.load_state_dict(torch.load('red_neuronal.pth'))
    model.eval()

    plt.figure(figsize=(10,5))
    plt.scatter(X, y, color='blue', label='Datos Reales')
    plt.plot(X, (model(torch.FloatTensor(X_scaled)).detach().numpy() * (y_max - y_min)) + y_min, color='red', label='Predicciones')
    plt.title('Predicciones vs Ventas Reales')
    plt.xlabel('Día del Mes')
    plt.ylabel('Ventas')
    plt.legend()
    st.pyplot(plt)

    ## Graficar Costo
    plt.figure(figsize=(10,5))
    plt.plot(loss_history)
    plt.title('Evolución del Costo')
    plt.xlabel('Épocas')
    plt.ylabel('Costo')
    plt.grid()
    st.pyplot(plt)