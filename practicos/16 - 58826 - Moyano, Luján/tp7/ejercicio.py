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

st.title('Estimativo de Ventas Diarias')

data = pd.read_csv('ventas.csv')
st.subheader('Datos de Ventas Diarias')
st.write(data)

scaler = MinMaxScaler()
data[['dia', 'ventas']] = scaler.fit_transform(data[['dia', 'ventas']])

st.sidebar.title('Parametros de la Red Neuronal')
learning_rate = st.sidebar.slider('Tasa de aprendizaje', 0.0, 1.0, 0.1, 0.01)
epochs = st.sidebar.slider('Cantidad de epocas', 10, 10000, 100, 10)
hidden_neurons = st.sidebar.slider('Neuronas en la capa oculta', 1, 100, 5, 1)
train_button = st.sidebar.button('Entrenar')


X = torch.tensor(data['dia'].values, dtype=torch.float32).view(-1, 1)
y = torch.tensor(data['ventas'].values, dtype=torch.float32).view(-1, 1)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

def train_model(model, dataloader, criterion, optimizer, epochs):
    progress_bar = st.progress(0)
    loss_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

       
        loss_history.append(running_loss / len(dataloader))
        
        
        progress_bar.progress((epoch + 1) / epochs)

    st.success('Entrenamiento completado de manera exitosa')
    return loss_history


fig, ax = plt.subplots()
ax.scatter(data['dia'], data['ventas'], color='blue', label='Datos reales')
ax.set_xlabel('dia')
ax.set_ylabel('Ventas')
st.pyplot(fig)


if train_button:

    input_size = 1
    output_size = 1
    model = NeuralNet(input_size, hidden_neurons, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    
    loss_history = train_model(model, dataloader, criterion, optimizer, epochs)
    

    fig2, ax2 = plt.subplots()
    ax2.plot(range(epochs), loss_history, color='orange')
    ax2.set_title('Evolucion de la Funcion de Costo')
    ax2.set_xlabel('Epoca')
    ax2.set_ylabel('Costo')
    st.pyplot(fig2)


    with torch.no_grad():
        predictions = model(X).numpy()
    
    fig3, ax3 = plt.subplots()
    ax3.scatter(data['dia'], data['ventas'], color='blue', label='Datos reales')
    ax3.plot(data['dia'], predictions, color='red', label='Predicción')
    ax3.set_xlabel('Dia')
    ax3.set_ylabel('Ventas')
    ax3.legend()
    st.pyplot(fig3)
