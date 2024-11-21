import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-parcial-medina-elias-manuel.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 59043')
        st.markdown('**Nombre:** Medina Elias Manuel')
        st.markdown('**Comisión:** C7')


## CARGA EL ARCHIVO CSV
def cargar_datos():
    try:
        with st.container():
            
            st.markdown("### 📊 Carga de Datos de Ventas")
            st.markdown("Por favor, sube tu archivo CSV con los datos de ventas")
            
        
        uploaded_file = st.file_uploader(
            "📂 Cargar archivo CSV de ventas",
            type=['csv'],
            help="El archivo debe contener las columnas: Sucursal, Producto, Año, Mes, Unidades_vendidas, Ingreso_total, Costo_total"
        )
        
        if uploaded_file is not None:
                with st.spinner('Procesando archivo...'):
                    df = pd.read_csv(uploaded_file)
                    st.success('✅ Archivo cargado exitosamente')
                    return df
        return None
            
    except Exception as e:
        st.error(f"❌ Error al cargar el archivo: {e}")
        return None
        

def calcular(datos, sucursal='Todas'):
    if sucursal != 'Todas':
        datos = datos[datos['Sucursal'] == sucursal]
    
    datos['Precio_unitario'] = datos['Ingreso_total'] / datos['Unidades_vendidas']
    
    resultados = datos.groupby('Producto').agg({
        'Unidades_vendidas': 'sum',      
        'Ingreso_total': 'sum',          
        'Costo_total': 'sum',            
        'Precio_unitario': 'mean'        
    }).reset_index()
    
    resultados = resultados.rename(columns={'Precio_unitario': 'Precio_promedio'})
    
    ganancia = resultados['Ingreso_total'] - resultados['Costo_total']
    resultados['Margen_promedio'] = (ganancia / resultados['Ingreso_total']) * 100
    resultados['Margen_promedio'] = resultados['Margen_promedio'].round(2)
    
    return resultados



def graficar_ventas(datos, sucursal='Todas'):
   
    if sucursal != 'Todas':
        datos = datos[datos['Sucursal'] == sucursal]
    
    figura, grafico = plt.subplots(figsize=(8, 6))
    
    ventas_mes = datos.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
    ventas_mes['Fecha'] = pd.to_datetime(ventas_mes['Año'].astype(str) + '-' + ventas_mes['Mes'].astype(str))
    ventas_mes = ventas_mes.sort_values('Fecha')
    
    grafico.plot(ventas_mes['Fecha'], ventas_mes['Unidades_vendidas'], 
                linestyle='-', label='Ventas reales')
    
    x = np.arange(len(ventas_mes))
    y = ventas_mes['Unidades_vendidas']
    tendencia = np.polyfit(x, y, 1) 
    linea_tendencia = np.poly1d(tendencia)
    grafico.plot(ventas_mes['Fecha'], linea_tendencia(x), 
                "r--", label='Tendencia')
    
    grafico.set_title('Evolución de Ventas en el Tiempo')
    grafico.set_xlabel('Fecha')
    grafico.set_ylabel('Cantidad Vendida')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    return figura

def calcular_cambio(datos, producto, metricas):
 
    datos_producto = datos[datos['Producto'] == producto].copy()
    
    datos_producto["Ingreso_por_unidad"] = datos_producto["Ingreso_total"] / datos_producto["Unidades_vendidas"]
    datos_producto["Porcentaje_margen"] = ((datos_producto["Ingreso_total"] - datos_producto["Costo_total"]) / datos_producto["Ingreso_total"]) * 100
    
    if metricas == 'Margen_promedio':
        metricas_por_año = datos_producto.groupby("Año")["Porcentaje_margen"].mean()
    elif metricas == 'Precio_promedio':
        metricas_por_año = datos_producto.groupby("Año")["Ingreso_por_unidad"].mean()
    else:  
        metricas_por_año = datos_producto.groupby("Año")["Unidades_vendidas"].sum()
    
    cambio_promedio = metricas_por_año.pct_change().mean() * 100
    
    return round(cambio_promedio, 2)



def main():
    st.title('Análisis de Ventas')
    mostrar_informacion_alumno()
    with st.sidebar:
        df = cargar_datos()
        if df is None:
            return
        
        sucursales = ['Todas'] + list(df['Sucursal'].unique())
        selecciona_sucursal = st.selectbox('Seleccione una sucursal:', sucursales)
    
    if df is not None:
        productos = df['Producto'].unique()
        
        for producto in productos:
            with st.container(border=True):
                st.subheader(f"📊 {producto}")
                
                col1, col2 = st.columns([1, 2])
                
                df_producto = df[df['Producto'] == producto]
                metricas = calcular(df_producto, selecciona_sucursal)
                
                cambioP = calcular_cambio(df, producto, 'Precio_promedio')
                cambioM = calcular_cambio(df, producto, 'Margen_promedio')
                cambioV = calcular_cambio(df, producto, 'Unidades_vendidas')
                
                with col1:
                    st.markdown("##### Métricas")
                    st.metric("💰 Precio Promedio", 
                            f"${metricas['Precio_promedio'].mean():.2f}", 
                            f"{cambioP:+.2f}%")
                    st.metric("📈 Margen Promedio", 
                            f"{metricas['Margen_promedio'].mean():.2f}%", 
                            f"{cambioM:+.2f}%")
                    st.metric("📦 Unidades Vendidas", 
                            f"{int(metricas['Unidades_vendidas'].sum()):,}", 
                            f"{cambioV:+.2f}%")
                
                with col2:
                    fig = graficar_ventas(df_producto, selecciona_sucursal)
                    st.pyplot(fig)

if __name__ == '__main__':
    main()