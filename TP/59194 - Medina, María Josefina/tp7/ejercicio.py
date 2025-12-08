import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim  


st.title("Prediccion de ventas diarias con REDES NEURONALES:")

st.sidebar.header("parametros de la red neuronal")
tasaAprendizaje = st.sidebar.slider("tasa de aprendizaje", 0.0, 1.0, 0.1)
epocas =st.sidebar.slider("cantidad de epocas", 10,10000,100)
neuronasOcultas = st.sidebar.slider("Neuronas en capa oculta", 1, 100, 5)
botonEntrenar = st.sidebar.button("ENTRENAR")

st.title("Estimacion de ventas diarias")

data = pd.read_csv("ventas.csv")
st.write("Datos de las ventas: ")
st.dataframe(data)
dias=data["dia"].values
ventas=data["ventas"].values

diasN = (dias- np.min(ventas)) / (np.max(ventas) - np.min (ventas))
diasT = torch.FloatTensor(diasN).view(-1,1)
plt.figure(figsize =(10,5))
plt.scatter(data["dia"], data["ventas"], color="blue", label = "ventas reales")
plt.xlabel("dia")
plt.title("ventas diarias")
st.pyplot(plt)


class modeloRed (nn.Module):
    def __init__(self, numNeuronas):
        super(modeloRed, self).__init__()
        
        self.capaOculta = nn.Linear(1, numNeuronas)
        self.capaSalida = nn.Linear(numNeuronas,1)
        
    def forward(self, x):
        x = torch.relu(self.capaOculta(x))
        x = self.capaSalida(x)
        return x

if botonEntrenar:
    x= torch.tensor(data["dia"].values, dtype=torch.float32).view(-1,1)
    y=torch.tensor(data["ventas"].values, dtype=torch.float32).view(-1,1)
    
    modelo = modeloRed(neuronasOcultas)
    criterio = nn.MSELoss()
    optimizador = optim.SGD(modelo.parameters(), lr=tasaAprendizaje)
    
    progreso = st.progress(0)
    valoresError = []

    for epoca in range(epocas):
        optimizador.zero_grad()
        predicciones = modelo(x)
        error = criterio(predicciones, y)
        error.backward()
        optimizador.step()
        valoresError.append(error.item())
        progreso.progress((epoca + 1) / epocas)
    st.success("Entrenamiento completo")
    
    
    
    st.write("Evolución de la función de costo:")
    plt.figure(figsize=(10,5))
    plt.plot(valoresError, label ="evolucion de perdidas")
    plt.xlabel("epocas")
    plt.ylabel("perdidas")
    st.pyplot(plt)
    
    
    st.write("Predicción de la red neuronal:")
    with torch.no_grad():
        prediccion = modelo(x)
        plt.figure(figsize=(10,5))
        plt.scatter(data["dia"], data["ventas"], color="blue", label="datos originales")
        plt.plot(data["dia"], prediccion.numpy(), color="red", label="Prediccion")
        plt.xlabel("dia")
        plt.ylabel("ventas")
        plt.title("ventas/prediccion")
        plt.legend()
        st.pyplot(plt)

            
