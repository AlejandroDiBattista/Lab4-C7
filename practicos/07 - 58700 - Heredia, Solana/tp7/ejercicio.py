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
st.title('Estimación de Ventas Diarias')
@st.cache
def cargar_datos():
    datos = pd.read_csv("ventas.csv")
    return datos
datos = cargar_datos()
st.write("datos de ventas", datos)
def normalizar_datos(datos):
    dato['ventas_normalizadas'] = (dato['ventas'] - dato['ventas'].min()) / (dato['ventas'].max() - dato['ventas'].min())
    return datos
dato = normalizar_datos(datos)
st.write('Datos normalizados', dato)

class ventasNN(nn.Module):
    def __init__(self):
        super(ventasNN,self).__init__()
        self.fc1 = nn.Linear(1,10)
        self.fc2 = nn.Linear(10,5)
        self.fc3 = nn.Linear(5,1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
modelo = ventasNN()      