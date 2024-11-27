import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-59044.streamlit.app/'

def mostrar_grafico_producto(df, producto, sucursal):
    if sucursal != "Todas":
        f = df[df['Sucursal'] == sucursal]
    else:
        f = df
    ventas_producto = f.groupby(['Año', 'Mes', 'Producto'])['Unidades_vendidas'].sum().reset_index()
    data = ventas_producto[ventas_producto['Producto'] == producto]
    data['Periodo'] = data['Año'].astype(str) + '-' + data['Mes'].astype(str)
    fig, ax = plt.subplots(figsize=(10, 6.3)) 
    ax.plot(data['Periodo'], data['Unidades_vendidas'], label=producto, linestyle='-', color='blue')
    ax.set_title(f'Evolución ventas mensual', fontsize=14)
    ax.set_xlabel('Año-Mes', fontsize=12)
    ax.set_ylabel('Unidades Vendidas', fontsize=10)

    etiquetasX = [
    str(año) if mes in [1] else ''
    for año, mes in zip(data['Año'], data['Mes'])]

    ax.set_xticklabels(etiquetasX, rotation=0)
    x = np.arange(len(data))
    y = data['Unidades_vendidas'].values
    coef = np.polyfit(x, y, 1)
    poly = np.poly1d(coef)
    tendencia = poly(x)
    ax.plot(data['Periodo'], tendencia, label='Tendencia', color='r', linestyle='--', linewidth=2)
    ax.legend(title="Producto")
    ax.set_ylim(0,None)
    ax.grid(True)
    plt.tight_layout()
    return fig

st.sidebar.header('Cargar archivo de datos')
archivo = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])

if archivo is None:
    st.title('Por favor,sube un archivo CSV desde la barra lateral.')
    def mostrar_informacion_alumno():
        with st.container(border=True):
            st.markdown('**Legajo:** 59044')
            st.markdown('**Nombre:** Naranjo Sosa Lucas')
            st.markdown('**Comisión:** C7')

    mostrar_informacion_alumno()
else:
    df = pd.read_csv(archivo)
    
    sucursal = df['Sucursal'].values
    producto = df['Producto'].values
    año = df['Año'].values
    mes = df['Mes'].values
    uvendidas = df['Unidades_vendidas'].values
    itotal = df['Ingreso_total'].values
    ctotal = df['Costo_total'].values

    opciones = ["Todas", "Sucursal Norte", "Sucursal Centro", "Sucursal Sur"]
    seleccion = st.sidebar.selectbox("Seleccionar sucursal", opciones)
    st.title('Datos de Todas las Sucursales')
    if 'Todas' in seleccion:
        productos = df['Producto'].unique()
        for producto in productos:
            filtroproducto = (df['Producto'] == producto)
            if filtroproducto.any():
                df['Preciop'] = df.loc[filtroproducto, 'Ingreso_total'] / df.loc[filtroproducto, 'Unidades_vendidas']
                preciop = df['Preciop'].mean()
                panual = df.groupby('Año')['Preciop'].mean()
                vprecio = panual.pct_change().mean() * 100
                df['ingcos'] = df.loc[filtroproducto, 'Ingreso_total'] - df.loc[filtroproducto, 'Costo_total']
                df['mar'] = (df['ingcos'] / df.loc[filtroproducto, 'Ingreso_total']) * 100
                margenpromedio_porcentaje = df['mar'].mean()
                margenanual = df.groupby('Año')['mar'].mean()
                vmargen = margenanual.pct_change().mean() * 100
                uanual=df.loc[filtroproducto].groupby('Año')['Unidades_vendidas'].sum()
                vunid=uanual.pct_change().mean() * 100   
                uvendidas_producto = df.loc[filtroproducto, 'Unidades_vendidas'].sum()   
            
            with st.container(border=True):
                st.subheader(f"{producto}")
                col1, col2 = st.columns([0.25, 0.75])
                with col1:
                    st.metric("Precio Promedio", f"${preciop:,.0f}".replace(",", "."),f"{vprecio:,.2f}%")
                    st.metric("Margen Promedio", f"{margenpromedio_porcentaje:.0f}%",f"{vmargen:,.2f}%")
                    st.metric("Unidades Vendidas", f"{uvendidas_producto:,.0f}".replace(",", "."),f"{vunid:,.2f}%")
                with col2:
                    fig = mostrar_grafico_producto(df, producto, seleccion)
                    st.pyplot(fig)

    if 'Sucursal Norte' in seleccion:
        productos = df['Producto'].unique()
        for producto in productos:
            filtrosucursal = (df['Sucursal'] == 'Sucursal Norte')
            filtroproductosolo = (df['Producto'] == producto)
            filtroproducto = filtrosucursal & filtroproductosolo
            if filtroproducto.any():
                df['Preciop'] = df.loc[filtroproducto, 'Ingreso_total'] / df.loc[filtroproducto, 'Unidades_vendidas']
                preciop = df['Preciop'].mean()
                panual = df.groupby('Año')['Preciop'].mean()
                vprecio = panual.pct_change().mean() * 100
                df['ingcos'] = df.loc[filtroproducto, 'Ingreso_total'] - df.loc[filtroproducto, 'Costo_total']
                df['mar'] = (df['ingcos'] / df.loc[filtroproducto, 'Ingreso_total']) * 100
                margenpromedio_porcentaje = df['mar'].mean()
                margenanual = df.groupby('Año')['mar'].mean()
                vmargen = margenanual.pct_change().mean() * 100
                uanual=df.loc[filtroproducto].groupby('Año')['Unidades_vendidas'].sum()
                vunid=uanual.pct_change().mean() * 100   
                uvendidas_producto = df.loc[filtroproducto, 'Unidades_vendidas'].sum()    
            with st.container(border=True):
                st.subheader(f"{producto}")
                col1, col2 = st.columns([0.25, 0.75])
                with col1:
                    st.metric("Precio Promedio", f"${preciop:,.0f}".replace(",", "."),f"{vprecio:,.2f}%")
                    st.metric("Margen Promedio", f"{margenpromedio_porcentaje:.0f}%",f"{vmargen:,.2f}%")
                    st.metric("Unidades Vendidas", f"{uvendidas_producto:,.0f}".replace(",", "."),f"{vunid:,.2f}%")
                with col2:
                    fig = mostrar_grafico_producto(df, producto, seleccion)
                    st.pyplot(fig)

    if 'Sucursal Centro' in seleccion:
        productos = df['Producto'].unique()
        for producto in productos:
            filtrosucursal = (df['Sucursal'] == 'Sucursal Centro')
            filtroproductosolo = (df['Producto'] == producto)
            filtroproducto = filtrosucursal & filtroproductosolo
            if filtroproducto.any():
                df['Preciop'] = df.loc[filtroproducto, 'Ingreso_total'] / df.loc[filtroproducto, 'Unidades_vendidas']
                preciop = df['Preciop'].mean()
                panual = df.groupby('Año')['Preciop'].mean()
                vprecio = panual.pct_change().mean() * 100
                df['ingcos'] = df.loc[filtroproducto, 'Ingreso_total'] - df.loc[filtroproducto, 'Costo_total']
                df['mar'] = (df['ingcos'] / df.loc[filtroproducto, 'Ingreso_total']) * 100
                margenpromedio_porcentaje = df['mar'].mean()
                margenanual = df.groupby('Año')['mar'].mean()
                vmargen = margenanual.pct_change().mean() * 100
                uanual=df.loc[filtroproducto].groupby('Año')['Unidades_vendidas'].sum()
                vunid=uanual.pct_change().mean() * 100   
                uvendidas_producto = df.loc[filtroproducto, 'Unidades_vendidas'].sum()   
            with st.container(border=True):
                st.subheader(f"{producto}")
                col1, col2 = st.columns([0.25, 0.75])
                with col1:
                    st.metric("Precio Promedio", f"${preciop:,.0f}".replace(",", "."),f"{vprecio:,.2f}%")
                    st.metric("Margen Promedio", f"{margenpromedio_porcentaje:.0f}%",f"{vmargen:,.2f}%")
                    st.metric("Unidades Vendidas", f"{uvendidas_producto:,.0f}".replace(",", "."),f"{vunid:,.2f}%")
                with col2:
                    fig = mostrar_grafico_producto(df, producto, seleccion)
                    st.pyplot(fig)

    if 'Sucursal Sur' in seleccion:
        productos = df['Producto'].unique()
        for producto in productos:
            filtrosucursal = (df['Sucursal'] == 'Sucursal Sur')
            filtroproductosolo = (df['Producto'] == producto)
            filtroproducto = filtrosucursal & filtroproductosolo
            if filtroproducto.any():
                df['Preciop'] = df.loc[filtroproducto, 'Ingreso_total'] / df.loc[filtroproducto, 'Unidades_vendidas']
                preciop = df['Preciop'].mean()
                panual = df.groupby('Año')['Preciop'].mean()
                vprecio = panual.pct_change().mean() * 100
                df['ingcos'] = df.loc[filtroproducto, 'Ingreso_total'] - df.loc[filtroproducto, 'Costo_total']
                df['mar'] = (df['ingcos'] / df.loc[filtroproducto, 'Ingreso_total']) * 100
                margenpromedio_porcentaje = df['mar'].mean()
                margenanual = df.groupby('Año')['mar'].mean()
                vmargen = margenanual.pct_change().mean() * 100
                uanual=df.loc[filtroproducto].groupby('Año')['Unidades_vendidas'].sum()
                vunid=uanual.pct_change().mean() * 100   
                uvendidas_producto = df.loc[filtroproducto, 'Unidades_vendidas'].sum()   
            with st.container(border=True):
                st.subheader(f"{producto}")
                col1, col2 = st.columns([0.25, 0.75])
                with col1:
                    st.metric("Precio Promedio", f"${preciop:,.0f}".replace(",", "."),f"{vprecio:,.2f}%")
                    st.metric("Margen Promedio", f"{margenpromedio_porcentaje:.0f}%",f"{vmargen:,.2f}%")
                    st.metric("Unidades Vendidas", f"{uvendidas_producto:,.0f}".replace(",", "."),f"{vunid:,.2f}%")
                with col2:
                    fig = mostrar_grafico_producto(df, producto, seleccion)
                    st.pyplot(fig)

