import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

st.title('Estimación de Ventas Diarias')
datos = pd.read_csv('ventas.csv')
st.subheader('Datos de Ventas Diarias')
st.write(datos)



escalador = MinMaxScaler()
datos[['dia', 'ventas']] = escalador.fit_transform(datos[['dia', 'ventas']])
st.sidebar.title('Parámetros de la Red Neuronal')
tasa_aprendizaje = st.sidebar.slider('Tasa de aprendizaje', 0.0, 1.0, 0.1, 0.01)
epocas = st.sidebar.slider('Cantidad de épocas', 10, 10000, 100, 10)
neuronas_ocultas = st.sidebar.slider('Neuronas en la capa oculta', 1, 100, 5, 1)
boton_entrenar = st.sidebar.button('Entrenar')
X = torch.tensor(datos['dia'].values, dtype=torch.float32).view(-1, 1)
y = torch.tensor(datos['ventas'].values, dtype=torch.float32).view(-1, 1)
conjunto_datos = TensorDataset(X, y)
cargador_datos = DataLoader(conjunto_datos, batch_size=10, shuffle=True)



class RedNeuronal(nn.Module):
    def __init__(self, entrada_tam, oculto_tam, salida_tam):
        super(RedNeuronal, self).__init__()
        self.oculta = nn.Linear(entrada_tam, oculto_tam)
        self.relu = nn.ReLU()
        self.salida = nn.Linear(oculto_tam, salida_tam)
        
    def forward(self, x):
        x = self.oculta(x)
        x = self.relu(x)
        x = self.salida(x)
        return x



def entrenar_modelo(modelo, cargador_datos, criterio, optimizador, epocas):
    barra_progreso = st.progress(0)
    historial_perdida = []

    for epoca in range(epocas):
        perdida_acumulada = 0.0
        for entradas, objetivos in cargador_datos:
            optimizador.zero_grad()
            salidas = modelo(entradas)
            perdida = criterio(salidas, objetivos)
            perdida.backward()
            optimizador.step()
            perdida_acumulada += perdida.item()

        historial_perdida.append(perdida_acumulada / len(cargador_datos))
        barra_progreso.progress((epoca + 1) / epocas)

    st.success('Entrenamiento completado exitosamente')
    return historial_perdida



fig, ax = plt.subplots()
ax.scatter(datos['dia'], datos['ventas'], color='blue', label='Datos reales')
ax.set_xlabel('Día')
ax.set_ylabel('Ventas')
st.pyplot(fig)



if boton_entrenar:
    entrada_tam = 1
    salida_tam = 1
    modelo = RedNeuronal(entrada_tam, neuronas_ocultas, salida_tam)
    criterio = nn.MSELoss()
    optimizador = optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)
    historial_perdida = entrenar_modelo(modelo, cargador_datos, criterio, optimizador, epocas)
    fig2, ax2 = plt.subplots()
    ax2.plot(range(epocas), historial_perdida, color='orange')
    ax2.set_title('Evolución de la Función de Costo')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Costo')
    st.pyplot(fig2)



    with torch.no_grad():
        predicciones = modelo(X).numpy()
    
    fig3, ax3 = plt.subplots()
    ax3.scatter(datos['dia'], datos['ventas'], color='blue', label='Datos reales')
    ax3.plot(datos['dia'], predicciones, color='red', label='Predicción')
    ax3.set_xlabel('Día')
    ax3.set_ylabel('Ventas')
    ax3.legend()
    st.pyplot(fig3)
