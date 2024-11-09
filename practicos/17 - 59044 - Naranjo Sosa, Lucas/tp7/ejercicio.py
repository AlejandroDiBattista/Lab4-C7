import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

st.title('Estimación de Ventas Diarias')

st.sidebar.header('Parametros de Entrenamiento')
learning_rate = st.sidebar.number_input(
    'Tasa de aprendizaje', min_value=0.0, max_value=1.0, value=0.1, step=0.1
)
epocas = st.sidebar.number_input(
    'Epocas', min_value=10, max_value=10000, value=100, step=10
)
neuocultas = st.sidebar.number_input(
    'Neuronas capa oculta', min_value=1, max_value=100, value=5, step=1
)
botonentrenar = st.sidebar.button('Entrenar')

class Ventas(nn.Module):
    def __init__(self, neuocultas):
        super(Ventas, self).__init__()
        self.hidden = nn.Linear(1, neuocultas)
        self.output = nn.Linear(neuocultas, 1)
        self.X_min, self.X_max = None, None
        self.y_min, self.y_max = None, None
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x
    
    def normalizar(self, X, y):
        self.X_min, self.X_max = X.min(), X.max()
        self.y_min, self.y_max = y.min(), y.max()
        Xnormalizado = (X - self.X_min) / (self.X_max - self.X_min)
        Ynormalizado = (y - self.y_min) / (self.y_max - self.y_min)
        return Xnormalizado, Ynormalizado

    def desnormalizar(self, predictions):
        return predictions * (self.y_max - self.y_min) + self.y_min

data = pd.read_csv('ventas.csv')
X = torch.tensor(data['dia'].values, dtype=torch.float32).view(-1, 1)
y = torch.tensor(data['ventas'].values, dtype=torch.float32).view(-1, 1)

model = Ventas(neuocultas)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

Xnormalizado, Ynormalizado = model.normalizar(X, y)

if botonentrenar:
    progress_bar = st.progress(0)
    losses = []
    for epoca in range(epocas):
        predictions = model(Xnormalizado)
        loss = criterion(predictions, Ynormalizado)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        progress_bar.progress((epoca + 1) / epocas)
        
        if (epoca + 1) % (epocas // 10) == 0 or epoca == epocas - 1:
            st.write(f"Epoca {epoca + 1}/{epocas} - Error: {loss.item():.6f}")
    
    st.success("Entrenamiento exitoso")

    torch.save(model.state_dict(), 'modeloRedneuronal.pth')
    st.write("Modelo guardado como 'modeloRedneuronal.pth'")

    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(losses, color='green', label='Perdidas')
    ax_loss.set_xlabel('Epoca')
    ax_loss.set_ylabel('Perdida')
    ax_loss.legend()
    st.pyplot(fig_loss)

    with torch.no_grad():
        predictions = model(Xnormalizado)
        predictions = model.desnormalizar(predictions)

    fig, ax = plt.subplots()
    ax.scatter(data['dia'], data['ventas'], color='blue', label='Datos Reales')
    ax.plot(data['dia'], predictions.numpy(), color='red', label='Curva de Ajuste')
    ax.set_xlabel('Día del Mes')
    ax.set_ylabel('Ventas')
    ax.legend()
    st.pyplot(fig)
