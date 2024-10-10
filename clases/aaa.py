import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title('Graficador de Datos de Excel')

# Permitir al usuario subir un archivo de Excel
archivo_subido = st.file_uploader("Elige un archivo Excel", type=["xlsx", "xls", 'csv'])

if archivo_subido is not None:
    # Leer el archivo Excel subido
    df = pd.read_csv    (archivo_subido)
    st.write('Datos del archivo:')
    st.write(df)

    # Verificar que el archivo tenga al menos dos columnas
    if df.shape[1] >= 2:
        # Tomar la primera y segunda columna como x e y
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]

        # Crear la figura y el eje
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Gráfica de X vs Y')

        # Mostrar la gráfica en Streamlit
        st.pyplot(fig)
    else:
        st.error("El archivo debe contener al menos dos columnas.")
