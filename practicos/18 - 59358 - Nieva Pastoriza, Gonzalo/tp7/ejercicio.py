import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv("c:/workspace/Lab4-C7/practicos/18 - 59358 - Nieva Pastoriza, Gonzalo/tp7/ventas.csv")
X = data['dia'].values.reshape(-1, 1)
y = data['ventas'].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

class NeuralNetwork(nn.Module):
    def __init__(self, hidden_neurons):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1, hidden_neurons)  
        self.fc2 = nn.Linear(hidden_neurons, 1) 

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

st.title("Clasificación de Ventas Diarias")
st.sidebar.header("Parámetros de la Red Neuronal")

learning_rate = st.sidebar.slider("Tasa de Aprendizaje", 0.0, 1.0, 0.1)
epochs = st.sidebar.slider("Cantidad de Épocas", 10, 10000, 100)
hidden_neurons = st.sidebar.slider("Neuronas en Capa Oculta", 1, 100, 5)

if st.sidebar.button("Entrenar"):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    model = NeuralNetwork(hidden_neurons)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    progress_bar = st.sidebar.progress(0)
    loss_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(torch.FloatTensor(X_train))
        loss = criterion(outputs, torch.FloatTensor(y_train))
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        progress_bar.progress((epoch + 1) / epochs)

    st.sidebar.success("Entrenamiento finalizado con éxito!")

    plt.figure(figsize=(10,5))
    plt.plot(loss_history)
    plt.title('Evolución de la Función de Costo')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.grid()
    st.sidebar.pyplot(plt)

    model.eval()
    with torch.no_grad():
        predictions = model(torch.FloatTensor(X_scaled)).numpy()

    plt.figure(figsize=(10,5))
    plt.scatter(X, y, color='blue', label='Datos Reales')
    plt.plot(X, scaler_y.inverse_transform(predictions), color='red', label='Predicciones')
    plt.title('Predicciones vs Ventas Reales')
    plt.xlabel('Día del Mes')
    plt.ylabel('Ventas')
    plt.legend()
    st.pyplot(plt)