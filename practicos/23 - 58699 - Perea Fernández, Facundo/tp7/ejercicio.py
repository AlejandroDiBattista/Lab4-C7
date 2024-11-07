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

st.sidebar.header("Parámetros de la Red Neuronal")


learning_rate = st.sidebar.slider("Tasa de aprendizaje", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

epochs = st.sidebar.slider("Cantidad de épocas", min_value=10, max_value=10000, value=100, step=10)

hidden_neurons = st.sidebar.slider("Cantidad de neuronas en la capa oculta", min_value=1, max_value=100, value=5, step=1)


if st.sidebar.button("Entrenar"):
    # La red neuronal iria aqui
    pass

st.header("Resultados")