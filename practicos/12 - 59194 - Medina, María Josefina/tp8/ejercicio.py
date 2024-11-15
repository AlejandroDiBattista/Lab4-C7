import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-555555.streamlit.app/'

st.set_page_config(page_title="parcial", layout="centered", initial_sidebar_state="auto")
#st.markdown(<style>.css-1d391kg {background-color: white ;color: black}</style>,unsafe_allow_html=True
#)

st.title("Por favor, selecciona el archivo CSV desde la barra lateal")
st.sidebar.title("Archivos de datos")

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 59.194')
        st.markdown('**Nombre:** Medina Maria Josefina')
        st.markdown('**Comisión:** C7')

mostrar_informacion_alumno()

#data_gaseosa= pd.read_csv("gaseosas.csv")
#data_vinos= pd.read_csv("vinos.csv")
sucursales=["Todas", "Sucursal Norte", "Sucursal Sur", "Sucursal Centro"]
archivo=None
datos=None


archivosCSV = st.sidebar.file_uploader("Sube un archivo csv", type=["csv"])
if archivosCSV:
    archivo = pd.read_csv(archivosCSV)
    st.success("Archivo cargado correctamente")
#else:
#    archivo=None

seleccionSucursal= st.sidebar.selectbox("Seleccione una sucursal: ", sucursales)
if archivo is not None:
    if seleccionSucursal == "Todas":
        title = "Datos de todas las sucursales"
        datos = archivo
    else:
        title = f"Datos de {seleccionSucursal}"
        datos =archivo[archivo["Sucursal"] == seleccionSucursal]
        
    st.header(title)
    st.write(datos)


    st.write ("Tabla de datos:")
    st.dataframe(datos)

    st.subheader("Calculos por Producto")
    productos= datos["Producto"].unique()
    resultados= []

    for producto in productos:
        datosP = datos[datos["Producto"]==producto]
        unidades= datosP["Unidades vendidas"].sum()
        ingresoTotal=datosP["Ingreso total"].sum()
        costoTotal=datosP["Costo TOtal"].sum()
        precioPromedio=ingresoTotal / unidades if unidades> 0 else 0
        #
        margenPromedio = (ingresoTotal - costoTotal) / ingresoTotal if ingresoTotal > 0 else 0
        resultados.append({
            "Producto":producto,
            "Unidades Vendidas": unidades,
            "Precio Promedio": round(precioPromedio, 2),
            "Margen Promedio": round(margenPromedio, 2)
        })
        
        resultadosDatos = pd.DataFrame(resultados)
        st.dataframe(resultadosDatos)