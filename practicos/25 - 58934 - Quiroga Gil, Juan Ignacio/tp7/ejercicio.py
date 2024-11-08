import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

st.title('Estimación de Ventas Diarias')

class VentasNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=5, output_size=1):
        super(VentasNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        return x

df = pd.read_csv('ventas.csv')
X = df['dia'].values
y = df['ventas'].values

def normalizar_datos(data):
    mean = data.mean()
    std = data.std()
    return (data - mean) / std, mean, std

st.sidebar.header('Parámetros de Entrenamiento')

col1, col2 = st.sidebar.columns([3, 1])
with col1:
    st.text('Aprendizaje')
    learning_rate = st.number_input('', 
                                  min_value=0.0, 
                                  max_value=1.0, 
                                  value=0.01, 
                                  format='%0.4f',
                                  label_visibility='collapsed')


col3, col4 = st.sidebar.columns([3, 1])
with col3:
    st.text('Repeticiones')
    epochs = st.number_input('', 
                           min_value=10, 
                           max_value=10000, 
                           value=100,
                           step=10,
                           label_visibility='collapsed')


st.sidebar.text('Neuronas Capa Oculta')
col5, col6 = st.sidebar.columns([3, 1])
with col5:
    hidden_neurons = st.number_input('', 
                                   min_value=1, 
                                   max_value=100, 
                                   value=5,
                                   label_visibility='collapsed')

X_norm, X_mean, X_std = normalizar_datos(X)
y_norm, y_mean, y_std = normalizar_datos(y)

X_tensor = torch.FloatTensor(X_norm.reshape(-1, 1))
y_tensor = torch.FloatTensor(y_norm.reshape(-1, 1))

model = VentasNet(hidden_size=hidden_neurons)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if st.sidebar.button('Entrenar'):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    losses = []
    for epoch in range(epochs):
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f'Epoch {epoch+1}/{epochs} - Error: {loss.item():.6f}')
    
    st.sidebar.success('Entrenamiento exitoso')
    
    fig_loss, ax_loss = plt.subplots(figsize=(6, 2))
    ax_loss.plot(losses, color='green', label='Pérdidas')
    ax_loss.set_xlabel('Época')
    ax_loss.set_ylabel('Pérdida')
    ax_loss.grid(True)
    st.sidebar.pyplot(fig_loss)

with torch.no_grad():
    X_test = torch.FloatTensor(np.linspace(X_norm.min(), X_norm.max(), 100).reshape(-1, 1))
    y_pred_norm = model(X_test)
    
    X_test_denorm = X_test.numpy() * X_std + X_mean
    y_pred_denorm = y_pred_norm.numpy() * y_std + y_mean

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X, y, color='blue', label='Datos Reales', s=50)
ax.plot(X_test_denorm, y_pred_denorm, color='red', label='Curva de Ajuste', linewidth=2)
ax.set_xlabel('Día del Mes')
ax.set_ylabel('Ventas')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
ax.grid(True)
st.pyplot(fig)