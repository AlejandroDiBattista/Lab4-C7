import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class RedNeuronalVentas(nn.Module):
    def __init__(self, neuronas_ocultas):
        super(RedNeuronalVentas, self).__init__()
        self.capa_entrada = nn.Linear(1, neuronas_ocultas)
        self.capa_intermedia = nn.Linear(neuronas_ocultas, neuronas_ocultas)
        self.capa_salida = nn.Linear(neuronas_ocultas, 1)
        
    def forward(self, x):
        x = torch.relu(self.capa_entrada(x))
        x = torch.relu(self.capa_intermedia(x))
        return self.capa_salida(x)

def preparar_datos(ruta_archivo='ventas.csv'):
    """Cargar y preparar datos para entrenamiento"""
    df = pd.read_csv(ruta_archivo)
    
    normalizador_x = MinMaxScaler()
    normalizador_y = MinMaxScaler()
    
    x_normalized = normalizador_x.fit_transform(df['dia'].values.reshape(-1, 1))
    y_normalized = normalizador_y.fit_transform(df['ventas'].values.reshape(-1, 1))
    
    x_tensor = torch.FloatTensor(x_normalized)
    y_tensor = torch.FloatTensor(y_normalized)
    
    return x_tensor, y_tensor, normalizador_x, normalizador_y, df

def entrenar_red_neuronal(x, y, tasa_aprendizaje, epocas, neuronas_ocultas, barra_progreso=None, mensaje_estado=None):
    modelo = RedNeuronalVentas(neuronas_ocultas)
    criterio = nn.MSELoss()
    optimizador = optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)
    
    historial_perdida = []
    
    for epoca in range(epocas):
        optimizador.zero_grad()
        predicciones = modelo(x)
        perdida = criterio(predicciones, y)
        perdida.backward()
        optimizador.step()
        
        historial_perdida.append(perdida.item())
        
        if barra_progreso is not None:
            barra_progreso.progress((epoca + 1) / epocas)
            mensaje_estado.text(f'Entrenando... Época {epoca+1}/{epocas}')
    
    return modelo, historial_perdida

def main():
    st.title('Predicción de Ventas con Red Neuronal')
    
    x, y, normalizador_x, normalizador_y, df = preparar_datos()
    
    st.sidebar.header('Configuración de Entrenamiento')
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        tasa_aprendizaje = st.number_input(
            'Aprendizaje', 
            min_value=0.0, 
            max_value=1.0, 
            value=0.1, 
            step=0.01
        )
    
    with col2:
        epocas = st.number_input(
            'Repeticiones', 
            min_value=10, 
            max_value=10000, 
            value=100,
            step=10
        )
    
    neuronas_ocultas = st.sidebar.number_input(
        'Neuronas en Capa Oculta', 
        min_value=1, 
        max_value=100, 
        value=5,
        step=1
    )
    
   
    boton_entrenar = st.sidebar.button('Entrenar')
    
   
    barra_progreso = st.sidebar.empty()
    mensaje_estado = st.sidebar.empty()
    
    if boton_entrenar:
      
        modelo, historial_perdida = entrenar_red_neuronal(
            x, y, tasa_aprendizaje, epocas, neuronas_ocultas,
            barra_progreso, mensaje_estado
        )
        
        mensaje_estado.success('Entrenamiento exitoso')
        
        fig1, ax1 = plt.subplots(figsize=(5, 2))
        ax1.plot(historial_perdida)
        ax1.set_title('Evolución de la Pérdida')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Pérdida')
        ax1.grid(True)
        st.sidebar.pyplot(fig1)
        
        with st.container():
            st.markdown("""
                <style>
                    .stContainer {
                        border: 2px solid #cccccc;
                        border-radius: 5px;
                        padding: 20px;
                        margin-bottom: 20px;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            with torch.no_grad():
                predicciones_norm = modelo(x).numpy()
                predicciones = normalizador_y.inverse_transform(predicciones_norm)
                dias = normalizador_x.inverse_transform(x.numpy())
                
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.scatter(dias, df['ventas'], label='Datos reales')
                ax2.plot(dias, predicciones, 'r', label='Predicciones')
                ax2.set_title('Predicción de Ventas')
                ax2.set_xlabel('Día')
                ax2.set_ylabel('Ventas')
                ax2.grid(True)
                ax2.legend()
                st.pyplot(fig2)

if __name__ == '__main__':
    main()