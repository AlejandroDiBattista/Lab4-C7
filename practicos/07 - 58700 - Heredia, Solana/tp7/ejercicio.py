import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


## Crear Red Neuronal
## Leer Datos
## Normalizar Datos
## Entrenar Red Neuronal
## Guardar Modelo
## Graficar Predicciones
tasa_aprendizaje = st.sidebar.slider('Tasa de aprendizaje',0.0, 1.0, 0.1)
epocas = st.sidebar.slider('Cantidad de epocas', 1, 100000, 100)
neuronas_capa_oculta = st.sidebar.slider('Neuronas en la capa oculta', 1, 100, 5)

entrenar_btn = st.sidebar.button('Entrenar')

st.title('Estimación de Ventas Diarias')

datos = pd.read_csv('ventas.csv')
st.write("Datos de ventas")
st.dataframe(datos)
dias = datos['dia'].values
ventas = datos['ventas'].values

dias_normalizados = (dias - np.min(dias)) / (np.max(dias) - np.min(dias))
ventas_normalizadas = (ventas - np.min(ventas)) / (np.max(ventas) - np.min(ventas))

dias_tensor = torch.FloatTensor(dias_normalizados).view(-1, 1)
ventas_tensor = torch.FloatTensor(ventas_normalizadas).view(-1, 1)

plt.figure(figsize=(10, 5))
plt.scatter(datos['dia'], datos['ventas'], color='blue', label='Datos de ventas')
plt.xlabel('Dia')
plt.ylabel('Ventas')
plt.title('Ventas diarias')
st.pyplot(plt)

class RedNeuronal(nn.Module):
    def __init__(self, num_neuronas):
        super(RedNeuronal, self).__init__()
        self.fc1 = nn.Linear(1, num_neuronas) #capa entrada
        self.fc2 = nn.Linear(num_neuronas, 1)# capa salida

    def forward(self, x):
        x = torch.relu(self.fc1(x))# app
        x = self.fc2(x)# salida
        return x
    
if entrenar_btn:
    x = torch.tensor(datos['dia'].values, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(datos['ventas'].values, dtype=torch.float32).view(-1, 1)

    modelo = RedNeuronal(neuronas_capa_oculta)
    criterio = nn.MSELoss()
    optimizador = optim.SGD(modelo.parameters(), lr=tasa_aprendizaje)

    progreso = st.progress(0)
    perdidas = []

    for epoca in range(epocas):
        predicciones = modelo(x)
        perdida = criterio(predicciones, y)
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()

        perdidas.append(perdida.item())
        progreso.progress((epoca + 1) / epocas)

    st.success('Entrenamiento finalizado')

    # Gráfico de la función de costo
    plt.figure(figsize=(10, 5))
    plt.plot(perdidas, label='Evolución de la pérdida')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Evolución de la función de costo')
    st.pyplot(plt)
    
    with torch.no_grad():
        prediccion = modelo(x)
        plt.figure(figsize=(10, 5))
        plt.scatter(datos['dia'], datos['ventas'], color='blue', label='Datos originales')
        plt.plot(datos['dia'], prediccion.numpy(), color='red', label='Predicción de la red')
        plt.xlabel('Dia')
        plt.ylabel('Ventas')
        plt.title('Ventas vs Predicción')
        plt.legend()
        st.pyplot(plt)
