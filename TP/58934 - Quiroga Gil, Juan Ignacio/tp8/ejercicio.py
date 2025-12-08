import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58934-jiqg.streamlit.app'
def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58934')
        st.markdown('**Nombre:** Juan Ignacio Quiroga Gil')
        st.markdown('**Comisión:** C7')
def hacer_grafico(d, p):
    ventas = d.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
    f, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(ventas)), ventas['Unidades_vendidas'], color='#ff5aff', label=p)
    x = np.arange(len(ventas))
    y = ventas['Unidades_vendidas']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), linestyle='-', color='#3c3c3c', label='Tendencia')
    ax.set_title('Evolución de Ventas Mensual')
    ax.set_xlabel('Año-Mes')
    ax.set_xticks(range(len(ventas)))
    lbls = []
    for i, row in enumerate(ventas.itertuples()):
        if row.Mes == 1:
            lbls.append(f"{row.Año}")
        else:
            lbls.append("")
    ax.set_xticklabels(lbls)
    ax.set_ylabel('Unidades Vendidas')
    ax.set_ylim(0, None)
    ax.legend(title='Producto')
    ax.grid(True)
    return f
st.sidebar.header("Cargar archivo .csv")
file = st.sidebar.file_uploader("Subir archivo .csv", type=["csv"])
if file is not None:
    df = pd.read_csv(file)
    cambiar_branch = ["Todas"] + df['Sucursal'].unique().tolist()
    branch = st.sidebar.selectbox("Seleccionar Sucursal", cambiar_branch)
    if branch != "Todas":
        df = df[df['Sucursal'] == branch]
        st.title(f"Datos de {branch}")
    else:
        st.title("Datos de Todas las Sucursales")
    productos = df['Producto'].unique()
    for p in productos:
        with st.container(border=True):
            st.subheader(f"{p}")
            temp = df[df['Producto'] == p]
            temp['price'] = temp['Ingreso_total'] / temp['Unidades_vendidas']
            precio_promedio = temp['price'].mean()
            precio_x_año = temp.groupby('Año')['price'].mean()
            cambio_precio = precio_x_año.pct_change().mean() * 100
            temp['profit'] = temp['Ingreso_total'] - temp['Costo_total']
            temp['margin'] = (temp['profit'] / temp['Ingreso_total']) * 100
            margen_promedio = temp['margin'].mean()
            margen_x_año = temp.groupby('Año')['margin'].mean()
            cambio_margen = margen_x_año.pct_change().mean() * 100
            unidades_promedio = temp['Unidades_vendidas'].mean()
            unidades_totales = temp['Unidades_vendidas'].sum()
            unidades_x_año = temp.groupby('Año')['Unidades_vendidas'].sum()
            cambio_unidades = unidades_x_año.pct_change().mean() * 100
            c1, c2 = st.columns([0.25, 0.75])
            with c1:
                st.metric(label="Precio Promedio", value=f"${precio_promedio:,.0f}".replace(",", "."), delta=f"{cambio_precio:.2f}%")
                st.metric(label="Margen Promedio", value=f"{margen_promedio:.0f}%".replace(",", "."), delta=f"{cambio_margen:.2f}%")
                st.metric(label="Unidades Vendidas", value=f"{unidades_totales:,.0f}".replace(",", "."), delta=f"{cambio_unidades:.2f}%")
            with c2:
                fig = hacer_grafico(temp, p)
                st.pyplot(fig)
else:
    st.subheader("Cargue un archivo .csv en la barra lateral")
    mostrar_informacion_alumno()