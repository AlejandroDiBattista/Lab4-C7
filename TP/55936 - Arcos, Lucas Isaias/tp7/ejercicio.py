import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
##crear red neuronal
class RedNeuronal(nn.Module):
    def __init__(self, num_neuronas_ocultas):
        super(RedNeuronal, self).__init__()
        self.capa_oculta = nn.Linear(1, num_neuronas_ocultas)
        self.activacion = nn.ReLU()
        self.capa_salida = nn.Linear(num_neuronas_ocultas, 1)
    
    def forward(self, x):
        x = self.capa_oculta(x)
        x = self.activacion(x)
        x = self.capa_salida(x)
        return x

class DatasetVentas(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def preparar_datos():
    # Leer datos
    df = pd.read_csv('ventas.csv')
    
    # Normalizar datos
    X = df['dia'].values.reshape(-1, 1)
    y = df['ventas'].values.reshape(-1, 1)
    
    # Normalización Min-Max
    X_min, X_max = X.min(), X.max()
    y_min, y_max = y.min(), y.max()
    
    X_norm = (X - X_min) / (X_max - X_min)
    y_norm = (y - y_min) / (y_max - y_min)
    
    return X_norm, y_norm, X, y, (X_min, X_max), (y_min, y_max)

# entrenar red Neuronal
def entrenar_red(modelo, X_train, y_train, epocas, tasa_aprendizaje, progress_bar):
    criterio = nn.MSELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)
    
    dataset = DatasetVentas(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    perdidas = []
    
    for epoca in range(epocas):
        for X_batch, y_batch in dataloader:
           
            prediccion = modelo(X_batch)
            perdida = criterio(prediccion, y_batch)
            
         
            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()
        
        perdidas.append(perdida.item())
        progress_bar.progress((epoca + 1) / epocas)
    
    return perdidas


def main():
    st.title('Estimación de Ventas Diarias')
    

    st.sidebar.header('Parámetros de Entrenamiento')
    
    tasa_aprendizaje = st.sidebar.slider(
        'Tasa de aprendizaje',
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.001
    )
    
    epocas = st.sidebar.slider(
        'Épocas',
        min_value=10,
        max_value=10000,
        value=1000,
        step=10
    )
    
    num_neuronas = st.sidebar.slider(
        'Neuronas Capa Oculta',
        min_value=1,
        max_value=100,
        value=10,
        step=1
    )
    

    X_norm, y_norm, X, y, (X_min, X_max), (y_min, y_max) = preparar_datos()
    

    modelo = RedNeuronal(num_neuronas)
    
   
    if st.sidebar.button('Entrenar'):
        with st.spinner('Entrenando la red neuronal...'):
            progress_bar = st.progress(0)
            
          
            historico_perdidas = entrenar_red(
                modelo, X_norm, y_norm, epocas, tasa_aprendizaje, progress_bar
            )
            
            st.success('¡Entrenamiento completado!')
            
           
            fig_loss, ax_loss = plt.subplots()
            ax_loss.plot(historico_perdidas)
            ax_loss.set_xlabel('Época')
            ax_loss.set_ylabel('Pérdida')
            ax_loss.set_title('Evolución de la Función de Pérdida')
            st.pyplot(fig_loss)
            
       
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_norm)
                predicciones_norm = modelo(X_tensor)
                predicciones = predicciones_norm.numpy() * (y_max - y_min) + y_min
            
            # Graficar resultados
            fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
            ax_pred.scatter(X, y, c='blue', label='Datos Reales')
            ax_pred.plot(X, predicciones, 'r-', label='Curva de Ajuste')
            ax_pred.set_xlabel('Día del Mes')
            ax_pred.set_ylabel('Ventas')
            ax_pred.set_title('Estimación de Ventas Diarias')
            ax_pred.legend()
            st.pyplot(fig_pred)

if __name__ == '__main__':
    main()
st.title('Estimación de Ventas Diarias')