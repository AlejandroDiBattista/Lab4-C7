import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-59044.streamlit.app/'

def formato_color(valor):
    if valor > 0:
        return f'<span style="color: green;">↑ {valor:.2f}%</span>'
    elif valor < 0:
        return f'<span style="color: red;">↓ {valor:.2f}%</span>'
    else:
        return f'<span style="color: orange;">→ {valor:.2f}%</span>'


def mostrar_grafico_producto(df, producto, sucursal):
    if sucursal != "Todas":
        f = df[df['Sucursal'] == sucursal]
    else:
        f = df
    
    ventas_producto = f.groupby(['Año', 'Mes', 'Producto'])['Unidades_vendidas'].sum().reset_index()
    data = ventas_producto[ventas_producto['Producto'] == producto]
    data['Periodo'] = data['Año'].astype(str) + '-' + data['Mes'].astype(str)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10)) 
    ax.plot(data['Periodo'], data['Unidades_vendidas'], label=producto, linestyle='-', color='b')

    ax.set_title(f'Evolución ventas mensual', fontsize=16)
    ax.set_xlabel('Año-Mes', fontsize=12)
    ax.set_ylabel('Unidades Vendidas', fontsize=10)

    etiquetasX = [
    str(año) if mes in [1] else ''
    for año, mes in zip(data['Año'], data['Mes'])
]
    ax.set_xticklabels(etiquetasX, rotation=0)

    x = np.arange(len(data))
    y = data['Unidades_vendidas'].values
    coef = np.polyfit(x, y, 1)
    poly = np.poly1d(coef)
    tendencia = poly(x)

    ax.plot(data['Periodo'], tendencia, label='Tendencia', color='r', linestyle='--', linewidth=2)

    for i, periodo in enumerate(data['Periodo']):
        ax.axvline(x=i, color='gray', linestyle='--', linewidth=0.5)
    max_ventas = data['Unidades_vendidas'].max()
    for y in range(0, max_ventas, int(max_ventas / 5)):
        ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.7)

    ax.legend(title="Producto")
    ax.set_ylim(bottom=0)
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
                itotal_producto = df.loc[filtroproducto, 'Ingreso_total'].sum()
                uvendidas_producto = df.loc[filtroproducto, 'Unidades_vendidas'].sum()
                ctotal_producto = df.loc[filtroproducto, 'Costo_total'].sum()

                if uvendidas_producto > 0:
                    precio_promedio = itotal_producto / uvendidas_producto

                ua = df['Año'].max()
                um = df[df['Año'] == ua]['Mes'].max()
                
                if um == 1:
                    aa = ua - 1
                    am = 12
                else:
                    aa = ua
                    am = um - 1

                filtroanteriormes = (df['Mes'] == am)
                filtroanterioraño = (df['Año'] == aa)
                filtroactualmes = (df['Mes'] == um)
                filtroactualaño = (df['Año'] == ua)

                filtrotodosmesesanteriores = (df['Año'] < ua) | ((df['Año'] == ua) & (df['Mes'] < um))
                filtrocomanterior = filtroproducto & filtroanterioraño & filtroanteriormes
                filtrocomultimo = filtroproducto & filtroactualmes & filtroactualaño

                itotalultimomes = df.loc[filtrocomultimo, 'Ingreso_total'].sum()
                uvendidasultimomes = df.loc[filtrocomultimo, 'Unidades_vendidas'].sum()
                ctotalultimomes = df.loc[filtrocomultimo, 'Costo_total'].sum()

                if uvendidasultimomes > 0:
                    preciopromedioum = itotalultimomes / uvendidasultimomes
                    
                itotalanteriormes = df.loc[filtrocomanterior, 'Ingreso_total'].sum()
                uvendidasanteriormes = df.loc[filtrocomanterior, 'Unidades_vendidas'].sum()
                ctotalanteriormes = df.loc[filtrocomanterior, 'Costo_total'].sum()

                if uvendidasanteriormes > 0:
                    preciopromedioam = itotalanteriormes / uvendidasanteriormes

                cambioporcentualprom = ((preciopromedioum - preciopromedioam) / preciopromedioam) * 100
                if itotal_producto > 0:
                    margenpromedio = (itotal_producto - ctotal_producto) / itotal_producto
                    margenpromedio_porcentaje = margenpromedio * 100

                    margenpromedioanterior =( itotalanteriormes - ctotalanteriormes) / itotalanteriormes
                    margenpromedioanterior_porcentaje = margenpromedioanterior * 100

                    cambioporcentualmargenprom = ((margenpromedio_porcentaje - margenpromedioanterior_porcentaje) / margenpromedioanterior_porcentaje) * 100
                    
                cambioporcentualuvendidas = ((uvendidasultimomes - uvendidasanteriormes) / uvendidasanteriormes) * 100
            
            with st.container(border=True):
                col1, col2 = st.columns([0.3, 0.7])
                with col1:
                    st.markdown(f"<h1 style='font-size: 30px;'>{producto}</h1>", unsafe_allow_html=True)
                    st.markdown('Precio Promedio')
                    st.markdown(f"<h1 style='font-size: 40px;'>{precio_promedio:.0f}</h1>", unsafe_allow_html=True)
                    st.markdown(f"{formato_color(cambioporcentualprom)}", unsafe_allow_html=True)
                    st.markdown('Margen Promedio')
                    st.markdown(f"<h1 style='font-size: 40px;'>{margenpromedio_porcentaje:.0f}%</h1>", unsafe_allow_html=True)
                    st.markdown(f"{formato_color(cambioporcentualmargenprom)}", unsafe_allow_html=True)
                    st.markdown('Unidades Vendidas')
                    st.markdown(f"<h1 style='font-size: 40px;'>{uvendidas_producto}</h1>", unsafe_allow_html=True)
                    st.markdown(f"{formato_color(cambioporcentualuvendidas)}", unsafe_allow_html=True)
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
                itotal_producto = df.loc[filtroproducto, 'Ingreso_total'].sum()
                uvendidas_producto = df.loc[filtroproducto, 'Unidades_vendidas'].sum()
                ctotal_producto = df.loc[filtroproducto, 'Costo_total'].sum()

                if uvendidas_producto > 0:
                    precio_promedio = itotal_producto / uvendidas_producto
            
                ua = df['Año'].max()
                um = df[df['Año'] == ua]['Mes'].max()
                
                if um == 1:
                    aa = ua - 1
                    am = 12
                else:
                    aa = ua
                    am = um - 1

                filtroanteriormes = (df['Mes'] == am)
                filtroanterioraño = (df['Año'] == aa)
                filtroactualmes = (df['Mes'] == um)
                filtroactualaño = (df['Año'] == ua)

                filtrotodosmesesanteriores = (df['Año'] < ua) | ((df['Año'] == ua) & (df['Mes'] < um))
                filtrocomanterior = filtroproducto & filtroanterioraño & filtroanteriormes
                filtrocomultimo = filtroproducto & filtroactualmes & filtroactualaño
                itotalultimomes = df.loc[filtrocomultimo, 'Ingreso_total'].sum()
                uvendidasultimomes = df.loc[filtrocomultimo, 'Unidades_vendidas'].sum()
                ctotalultimomes = df.loc[filtrocomultimo, 'Costo_total'].sum()

                if uvendidasultimomes > 0:
                    preciopromedioum = itotalultimomes / uvendidasultimomes
                itotalanteriormes = df.loc[filtrocomanterior, 'Ingreso_total'].sum()
                uvendidasanteriormes = df.loc[filtrocomanterior, 'Unidades_vendidas'].sum()
                ctotalanteriormes = df.loc[filtrocomanterior, 'Costo_total'].sum()

                if uvendidasanteriormes > 0:
                    preciopromedioam = itotalanteriormes / uvendidasanteriormes
                cambioporcentualprom = ((preciopromedioum - preciopromedioam) / preciopromedioam) * 100
                if itotal_producto > 0:
                    margenpromedio = (itotal_producto - ctotal_producto) / itotal_producto
                    margenpromedio_porcentaje = margenpromedio * 100

                    margenpromedioanterior =( itotalanteriormes - ctotalanteriormes) / itotalanteriormes
                    margenpromedioanterior_porcentaje = margenpromedioanterior * 100

                    cambioporcentualmargenprom = ((margenpromedio_porcentaje - margenpromedioanterior_porcentaje) / margenpromedioanterior_porcentaje) * 100
                    
            
            cambioporcentualuvendidas = ((uvendidasultimomes - uvendidasanteriormes) / uvendidasanteriormes) * 100
            with st.container(border=True):
                col1, col2 = st.columns([0.3, 0.7])
                with col1:
                    st.markdown(f"<h1 style='font-size: 30px;'>{producto}</h1>", unsafe_allow_html=True)
                    st.markdown('Precio Promedio')
                    st.markdown(f"<h1 style='font-size: 40px;'>{precio_promedio:.0f}</h1>", unsafe_allow_html=True)
                    st.markdown(f"{formato_color(cambioporcentualprom)}", unsafe_allow_html=True)
                    st.markdown('Margen Promedio')
                    st.markdown(f"<h1 style='font-size: 40px;'>{margenpromedio_porcentaje:.0f}%</h1>", unsafe_allow_html=True)
                    st.markdown(f"{formato_color(cambioporcentualmargenprom)}", unsafe_allow_html=True)
                    st.markdown('Unidades Vendidas')
                    st.markdown(f"<h1 style='font-size: 40px;'>{uvendidas_producto}</h1>", unsafe_allow_html=True)
                    st.markdown(f"{formato_color(cambioporcentualuvendidas)}", unsafe_allow_html=True)
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
                itotal_producto = df.loc[filtroproducto, 'Ingreso_total'].sum()
                uvendidas_producto = df.loc[filtroproducto, 'Unidades_vendidas'].sum()
                ctotal_producto = df.loc[filtroproducto, 'Costo_total'].sum()

                if uvendidas_producto > 0:
                    precio_promedio = itotal_producto / uvendidas_producto
            
                ua = df['Año'].max()
                um = df[df['Año'] == ua]['Mes'].max()
                
                if um == 1:
                    aa = ua - 1
                    am = 12
                else:
                    aa = ua
                    am = um - 1

                filtroanteriormes = (df['Mes'] == am)
                filtroanterioraño = (df['Año'] == aa)
                filtroactualmes = (df['Mes'] == um)
                filtroactualaño = (df['Año'] == ua)

                filtrotodosmesesanteriores = (df['Año'] < ua) | ((df['Año'] == ua) & (df['Mes'] < um))
                filtrocomanterior = filtroproducto & filtroanterioraño & filtroanteriormes
                filtrocomultimo = filtroproducto & filtroactualmes & filtroactualaño
                itotalultimomes = df.loc[filtrocomultimo, 'Ingreso_total'].sum()
                uvendidasultimomes = df.loc[filtrocomultimo, 'Unidades_vendidas'].sum()
                ctotalultimomes = df.loc[filtrocomultimo, 'Costo_total'].sum()

                if uvendidasultimomes > 0:
                    preciopromedioum = itotalultimomes / uvendidasultimomes

                itotalanteriormes = df.loc[filtrocomanterior, 'Ingreso_total'].sum()
                uvendidasanteriormes = df.loc[filtrocomanterior, 'Unidades_vendidas'].sum()
                ctotalanteriormes = df.loc[filtrocomanterior, 'Costo_total'].sum()

                if uvendidasanteriormes > 0:
                    preciopromedioam = itotalanteriormes / uvendidasanteriormes

                cambioporcentualprom = ((preciopromedioum - preciopromedioam) / preciopromedioam) * 100
                if itotal_producto > 0:
                    margenpromedio = (itotal_producto - ctotal_producto) / itotal_producto
                    margenpromedio_porcentaje = margenpromedio * 100

                    margenpromedioanterior =( itotalanteriormes - ctotalanteriormes) / itotalanteriormes
                    margenpromedioanterior_porcentaje = margenpromedioanterior * 100

                    cambioporcentualmargenprom = ((margenpromedio_porcentaje - margenpromedioanterior_porcentaje) / margenpromedioanterior_porcentaje) * 100
                    
            cambioporcentualuvendidas = ((uvendidasultimomes - uvendidasanteriormes) / uvendidasanteriormes) * 100
            with st.container(border=True):
                col1, col2 = st.columns([0.3, 0.7])
                with col1:
                    st.markdown(f"<h1 style='font-size: 30px;'>{producto}</h1>", unsafe_allow_html=True)
                    st.markdown('Precio Promedio')
                    st.markdown(f"<h1 style='font-size: 40px;'>{precio_promedio:.0f}</h1>", unsafe_allow_html=True)
                    st.markdown(f"{formato_color(cambioporcentualprom)}", unsafe_allow_html=True)
                    st.markdown('Margen Promedio')
                    st.markdown(f"<h1 style='font-size: 40px;'>{margenpromedio_porcentaje:.0f}%</h1>", unsafe_allow_html=True)
                    st.markdown(f"{formato_color(cambioporcentualmargenprom)}", unsafe_allow_html=True)
                    st.markdown('Unidades Vendidas')
                    st.markdown(f"<h1 style='font-size: 40px;'>{uvendidas_producto}</h1>", unsafe_allow_html=True)
                    st.markdown(f"{formato_color(cambioporcentualuvendidas)}", unsafe_allow_html=True)

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
                itotal_producto = df.loc[filtroproducto, 'Ingreso_total'].sum()
                uvendidas_producto = df.loc[filtroproducto, 'Unidades_vendidas'].sum()
                ctotal_producto = df.loc[filtroproducto, 'Costo_total'].sum()

                if uvendidas_producto > 0:
                    precio_promedio = itotal_producto / uvendidas_producto
            
                ua = df['Año'].max()
                um = df[df['Año'] == ua]['Mes'].max()
                
                if um == 1:
                    aa = ua - 1
                    am = 12
                else:
                    aa = ua
                    am = um - 1

                filtroanteriormes = (df['Mes'] == am)
                filtroanterioraño = (df['Año'] == aa)
                filtroactualmes = (df['Mes'] == um)
                filtroactualaño = (df['Año'] == ua)

                filtrotodosmesesanteriores = (df['Año'] < ua) | ((df['Año'] == ua) & (df['Mes'] < um))
                filtrocomanterior = filtroproducto & filtroanterioraño & filtroanteriormes
                filtrocomultimo = filtroproducto & filtroactualmes & filtroactualaño

                itotalultimomes = df.loc[filtrocomultimo, 'Ingreso_total'].sum()
                uvendidasultimomes = df.loc[filtrocomultimo, 'Unidades_vendidas'].sum()
                ctotalultimomes = df.loc[filtrocomultimo, 'Costo_total'].sum()

                if uvendidasultimomes > 0:
                    preciopromedioum = itotalultimomes / uvendidasultimomes
                    
                itotalanteriormes = df.loc[filtrocomanterior, 'Ingreso_total'].sum()
                uvendidasanteriormes = df.loc[filtrocomanterior, 'Unidades_vendidas'].sum()
                ctotalanteriormes = df.loc[filtrocomanterior, 'Costo_total'].sum()

                if uvendidasanteriormes > 0:
                    preciopromedioam = itotalanteriormes / uvendidasanteriormes

                cambioporcentualprom = ((preciopromedioum - preciopromedioam) / preciopromedioam) * 100
                if itotal_producto > 0:
                    margenpromedio = (itotal_producto - ctotal_producto) / itotal_producto
                    margenpromedio_porcentaje = margenpromedio * 100

                    margenpromedioanterior =( itotalanteriormes - ctotalanteriormes) / itotalanteriormes
                    margenpromedioanterior_porcentaje = margenpromedioanterior * 100

                    cambioporcentualmargenprom = ((margenpromedio_porcentaje - margenpromedioanterior_porcentaje) / margenpromedioanterior_porcentaje) * 100
                    
            
            cambioporcentualuvendidas = ((uvendidasultimomes - uvendidasanteriormes) / uvendidasanteriormes) * 100

            with st.container(border=True):
                col1, col2 = st.columns([0.3, 0.7])
                with col1:
                    st.markdown(f"<h1 style='font-size: 30px;'>{producto}</h1>", unsafe_allow_html=True)
                    st.markdown('Precio Promedio')
                    st.markdown(f"<h1 style='font-size: 40px;'>{precio_promedio:.0f}</h1>", unsafe_allow_html=True)
                    st.markdown(f"{formato_color(cambioporcentualprom)}", unsafe_allow_html=True)
                    st.markdown('Margen Promedio')
                    st.markdown(f"<h1 style='font-size: 40px;'>{margenpromedio_porcentaje:.0f}%</h1>", unsafe_allow_html=True)
                    st.markdown(f"{formato_color(cambioporcentualmargenprom)}", unsafe_allow_html=True)
                    st.markdown('Unidades Vendidas')
                    st.markdown(f"<h1 style='font-size: 40px;'>{uvendidas_producto}</h1>", unsafe_allow_html=True)
                    st.markdown(f"{formato_color(cambioporcentualuvendidas)}", unsafe_allow_html=True)
                with col2:
                    fig = mostrar_grafico_producto(df, producto, seleccion)
                    st.pyplot(fig)

