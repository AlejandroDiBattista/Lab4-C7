import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class ModeloPrediccion(nn.Module):
    def __init__(self, unidades_ocultas):
        super(ModeloPrediccion, self).__init__()
        self.entrada = nn.Linear(1, unidades_ocultas)
        self.oculta = nn.Linear(unidades_ocultas, unidades_ocultas)
        self.salida = nn.Linear(unidades_ocultas, 1)
        
    def forward(self, entrada):
        activacion = torch.relu(self.entrada(entrada))
        activacion = torch.relu(self.oculta(activacion))
        return self.salida(activacion)

def cargar_datos(ruta_datos='datos.csv'):
    """Cargar datos desde un archivo CSV y normalizar"""
    data = pd.read_csv(ruta_datos)
    
    escalador_x = MinMaxScaler()
    escalador_y = MinMaxScaler()
    
    x_norm = escalador_x.fit_transform(data['dia'].values.reshape(-1, 1))
    y_norm = escalador_y.fit_transform(data['ventas'].values.reshape(-1, 1))
    
    x_tensor = torch.FloatTensor(x_norm)
    y_tensor = torch.FloatTensor(y_norm)
    
    return x_tensor, y_tensor, escalador_x, escalador_y, data

def entrenar_modelo(x, y, lr, iteraciones, unidades_ocultas, barra=None, mensaje=None):
    red = ModeloPrediccion(unidades_ocultas)
    funcion_perdida = nn.MSELoss()
    optimizador = optim.Adam(red.parameters(), lr=lr)
    
    historial = []
    
    for iteracion in range(iteraciones):
        optimizador.zero_grad()
        salida = red(x)
        perdida = funcion_perdida(salida, y)
        perdida.backward()
        optimizador.step()
        
        historial.append(perdida.item())
        
        if barra is not None:
            barra.progress((iteracion + 1) / iteraciones)
            mensaje.text(f'Progreso: Iteración {iteracion + 1}/{iteraciones}')
    
    return red, historial

def app_principal():
    st.title('Predicción de Ventas Usando Redes Neuronales')
    
    x, y, escalador_x, escalador_y, data = cargar_datos()
    
    st.sidebar.header('Parámetros de Entrenamiento')
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        lr = st.number_input('Tasa de Aprendizaje', min_value=0.001, max_value=1.0, value=0.01, step=0.001)
    with col2:
        iteraciones = st.number_input('Iteraciones', min_value=10, max_value=10000, value=100, step=10)
    
    unidades_ocultas = st.sidebar.number_input('Unidades Ocultas', min_value=1, max_value=50, value=10, step=1)
    entrenar = st.sidebar.button('Iniciar Entrenamiento')
    
    barra_progreso = st.sidebar.empty()
    mensaje = st.sidebar.empty()
    
    if entrenar:
        modelo, historial = entrenar_modelo(x, y, lr, iteraciones, unidades_ocultas, barra_progreso, mensaje)
        mensaje.success('Entrenamiento Finalizado')
        
        fig1, ax1 = plt.subplots()
        ax1.plot(historial)
        ax1.set_title('Historial de Pérdida')
        ax1.set_xlabel('Iteración')
        ax1.set_ylabel('Pérdida')
        st.sidebar.pyplot(fig1)
        
        with torch.no_grad():
            predicciones_norm = modelo(x).numpy()
            predicciones = escalador_y.inverse_transform(predicciones_norm)
            dias = escalador_x.inverse_transform(x.numpy())
            
            fig2, ax2 = plt.subplots()
            ax2.scatter(dias, data['ventas'], label='Ventas Reales')
            ax2.plot(dias, predicciones, color='red', label='Predicción')
            ax2.set_title('Resultados de Predicción')
            ax2.set_xlabel('Día')
            ax2.set_ylabel('Ventas')
            ax2.legend()
            st.pyplot(fig2)

if __name__ == '__main__':
    app_principal()
