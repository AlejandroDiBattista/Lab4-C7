import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import streamlit as st

# Título de la aplicación Streamlit
st.title('Estimación de Ventas Diarias')

# Cargar datos desde un archivo CSV
df = pd.read_csv('ventas.csv')
dias = df['dia'].values
ventas = df['ventas'].values

# Normalizar los datos para el entrenamiento (escalar entre 0 y 1)
normalizador = MinMaxScaler()
dias_scaled = normalizador.fit_transform(dias.reshape(-1, 1))  # Normalización de los días
ventas_scaled = normalizador.fit_transform(ventas.reshape(-1, 1))  # Normalización de las ventas

# Sidebar para configurar los parámetros de entrenamiento
st.sidebar.header("Parámetros de Entrenamiento")
lr = st.sidebar.slider("Tasa de Aprendizaje", min_value=0.0, max_value=1.0, value=0.1)
epocas = st.sidebar.slider("Cantidad de Épocas", min_value=10, max_value=10000, value=100)
neuronas_ocultas = st.sidebar.slider("Neurona Capa Oculta", min_value=1, max_value=100, value=5)
entrenar_btn = st.sidebar.button("Entrenar")

# Definición de la red neuronal
class ModeloVentas(nn.Module):
    def __init__(self, neuronas_ocultas):
        super(ModeloVentas, self).__init__()
        self.capa_oculta = nn.Linear(1, neuronas_ocultas)  # Capa oculta
        self.capa_salida = nn.Linear(neuronas_ocultas, 1)  # Capa de salida

    def forward(self, x):
        x = torch.relu(self.capa_oculta(x))  # Función de activación ReLU
        x = self.capa_salida(x)
        return x

# Función para entrenar el modelo
def entrenar_modelo(lr, epocas, neuronas_ocultas):
    modelo = ModeloVentas(neuronas_ocultas)
    funcion_perdida = nn.MSELoss()  # Error cuadrático medio
    optimizador = optim.SGD(modelo.parameters(), lr=lr)  # Optimización por descenso de gradiente

    # Convertir los datos a tensores de PyTorch
    x_train = torch.tensor(dias_scaled, dtype=torch.float32)
    y_train = torch.tensor(ventas_scaled, dtype=torch.float32)

    errores = []

    # Mostrar progreso en Streamlit
    progreso_texto = st.empty()
    barra_progreso = st.progress(0)

    for epoca in range(epocas):
        optimizador.zero_grad()
        predicciones = modelo(x_train)
        error = funcion_perdida(predicciones, y_train)  # Calcular el error
        error.backward()  # Retropropagación del error
        optimizador.step()  # Actualización de los pesos

        errores.append(error.item())
        barra_progreso.progress((epoca + 1) / epocas)  # Actualizar la barra de progreso
        progreso_texto.text(f'Epoca {epoca + 1}/{epocas} - Error: {error.item():.6f}')

    return modelo, errores

# Entrenar el modelo si se presiona el botón
if entrenar_btn:
    modelo, errores = entrenar_modelo(lr, epocas, neuronas_ocultas)

    # Mostrar el gráfico de las pérdidas durante el entrenamiento
    fig, ax = plt.subplots()
    ax.plot(errores, color='green')
    ax.set_title("Pérdida durante el Entrenamiento")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida")
    st.pyplot(fig)

    # Realizar las predicciones con el modelo entrenado
    x_test = torch.tensor(dias_scaled, dtype=torch.float32)
    with torch.no_grad():  # Deshabilitar el cálculo de gradientes durante la predicción
        predicciones_scaled = modelo(x_test).numpy()

    # Desnormalizar las predicciones
    predicciones = normalizador.inverse_transform(predicciones_scaled)


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

    # Mostrar el gráfico con los datos reales y las predicciones
    fig, ax = plt.subplots()
    ax.scatter(dias, ventas, color='blue', label='Datos Reales')
    ax.plot(dias, predicciones, color='red', label='Predicciones')
    ax.set_title("Estimación de Ventas Diarias")
    ax.set_xlabel("Día del Mes")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)

    st.success("Entrenamiento completado con éxito")
