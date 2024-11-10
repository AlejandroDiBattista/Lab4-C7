import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

@st.cache
def load_data():
    df = pd.read_csv('ventas.csv')
    return df

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

def train_model(model, criterion, optimizer, X_train, y_train, epochs, learning_rate, progress_bar):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    
    loss_values = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.flatten(), y_train)
        loss.backward()
        optimizer.step()
        
        loss_values.append(loss.item())
        
        progress_bar.progress((epoch + 1) / epochs)
    
    return model, loss_values

def predict(model, X):
    model.eval()
    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X)
    return predictions.numpy()

st.sidebar.header('Parámetros de la Red Neuronal')

learning_rate = st.sidebar.slider("Tasa de Aprendizaje", 0.0, 1.0, 0.1)
epochs = st.sidebar.slider("Cantidad de Épocas", 10, 10000, 100)
neurons = st.sidebar.slider("Neuronas en la capa oculta", 1, 100, 5)

df = load_data()

scaler = MinMaxScaler()
X = df['dia'].values.reshape(-1, 1)  
y = df['ventas'].values 

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

input_size = 1 
output_size = 1 
hidden_size = neurons  

model = NeuralNetwork(input_size, hidden_size, output_size)
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

progress_bar = st.sidebar.progress(0)

if st.sidebar.button("Entrenar"):
    st.sidebar.text("Entrenando el modelo...")
    
    model, loss_values = train_model(model, criterion, optimizer, X_train, y_train, epochs, learning_rate, progress_bar)
    
    predictions = predict(model, X_scaled)
    
    st.sidebar.text("Entrenamiento completado exitosamente")
    
    st.subheader("Gráfico de Predicción vs Realidad")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df['dia'], df['ventas'], label='Ventas Reales', color='blue', marker='o')  # Usar puntos

    plt.plot(df['dia'], predictions, label='Predicciones de la Red Neuronal', color='red', linestyle='--')

    plt.xlabel('Dia')
    plt.ylabel('Ventas')
    plt.legend()
    st.pyplot(plt)

    plt.figure(figsize=(10, 6))
    plt.plot(loss_values)
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Evolución de la función de costo')
    st.pyplot(plt)
