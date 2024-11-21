import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCIÓN: Debe colocar la dirección en la que ha sido publicada la aplicación en la siguiente línea
# app_url = "https://tp8-55547.streamlit.app/"

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown("**Legajo:** 55547")
        st.markdown("**Nombre:** Cabrera Pablo Daniel")
        st.markdown("**Comisión:** C7")

def cargar_datos():
    try:
        st.write("Subir archivo CSV")
        archivo_csv = st.file_uploader("Drag and drop file here", type=["csv"])
        if archivo_csv is not None:
            datos = pd.read_csv(archivo_csv)
            columnas_necesarias = ["Año", "Mes", "Producto", "Unidades_vendidas", "Ingreso_total", "Costo_total"]
            columnas_faltantes = [col for col in columnas_necesarias if col not in datos.columns]
            if columnas_faltantes:
                st.error(f"Faltan las siguientes columnas: {columnas_faltantes}")
                return None
            
            datos["Fecha"] = pd.to_datetime(datos["Año"].astype(str) + "-" + datos["Mes"].astype(str).str.zfill(2) + "-01")
            return datos
        return None
    except Exception as error:
        st.error(f"Error al cargar los datos: {str(error)}")
        st.exception(error)
        return None

def graficar_tendencia_ventas(datos, producto_seleccionado):
    try:
        datos_producto = datos[datos["Producto"] == producto_seleccionado]
        st.write(f"Registros encontrados para {producto_seleccionado}: {len(datos_producto)}")
        
        if len(datos_producto) == 0:
            figura, eje = plt.subplots(figsize=(12, 6))  
            eje.text(0.5, 0.5, f"No hay datos disponibles para {producto_seleccionado}", 
                     horizontalalignment="center", verticalalignment="center")
            eje.set_axis_off()
            return figura
        
        ventas_mensuales = datos_producto.groupby("Fecha")["Unidades_vendidas"].sum().reset_index()
        
        if len(ventas_mensuales) < 2:
            figura, eje = plt.subplots(figsize=(12, 6)) 
            eje.text(0.5, 0.5, f"Datos insuficientes para graficar tendencia de {producto_seleccionado}", 
                     horizontalalignment="center", verticalalignment="center")
            eje.set_axis_off()
            return figura
        
        figura, eje = plt.subplots(figsize=(12, 5)) 
        eje.grid(True, linestyle="-", alpha=0.3)
        
        eje_x_numerico = np.arange(len(ventas_mensuales))
        
        eje.plot(ventas_mensuales["Fecha"], ventas_mensuales["Unidades_vendidas"], 
                 marker="o", label=producto_seleccionado, color="blue", linewidth=1.5)
        
        coeficientes = np.polyfit(eje_x_numerico, ventas_mensuales["Unidades_vendidas"], 1)
        polinomio = np.poly1d(coeficientes)
        eje.plot(ventas_mensuales["Fecha"], polinomio(eje_x_numerico), "r--", label="Tendencia", linewidth=1.5)
        
        eje.set_title("Evolución de Ventas Mensual", pad=20)
        eje.set_xlabel("Año-Mes")
        eje.set_ylabel("Unidades Vendidas")
        eje.legend()
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return figura
    
    except Exception as error:
        st.error(f"Error en graficar_tendencia_ventas: {str(error)}")
        st.exception(error)
        figura, eje = plt.subplots(figsize=(20, 15))
        eje.text(0.5, 0.5, f"Error al generar el gráfico: {str(error)}", 
                 horizontalalignment="center", verticalalignment="center")
        eje.set_axis_off()
        return figura

def iniciar_aplicacion():
    st.set_page_config(layout="wide")
    
    st.header("Información del Estudiante")
    mostrar_informacion_alumno()

    with st.sidebar:
        st.header("Cargar archivo de datos")
        datos_cargados = cargar_datos()
        
        if datos_cargados is not None:
            st.write("Seleccionar Sucursal")
            lista_sucursales = ["Todas"] + datos_cargados["Sucursal"].unique().tolist()
            sucursal_seleccionada = st.selectbox(label="Seleccionar Sucursal", 
                                                 options=lista_sucursales, 
                                                 label_visibility="collapsed")
    
    if datos_cargados is not None:
        st.header(f"Datos de {'Todas las Sucursales' if sucursal_seleccionada == 'Todas' else sucursal_seleccionada}")
        
        datos_filtrados = datos_cargados if sucursal_seleccionada == "Todas" else datos_cargados[datos_cargados["Sucursal"] == sucursal_seleccionada]
        
        lista_productos = datos_filtrados["Producto"].unique()
        
        for producto in lista_productos:
            with st.container(border=True):
                st.subheader(f"{producto}")
                datos_producto = datos_filtrados[datos_filtrados["Producto"] == producto]

                datos_producto["Precio_promedio"] = datos_producto["Ingreso_total"] / datos_producto["Unidades_vendidas"]
                promedio_precio = datos_producto["Precio_promedio"].mean()
                
                promedio_precio_anual = datos_producto.groupby("Año")["Precio_promedio"].mean()
                variacion_precio_anual = promedio_precio_anual.pct_change().mean() * 100
                
                datos_producto["Ganancia"] = datos_producto["Ingreso_total"] - datos_producto["Costo_total"]
                datos_producto["Margen"] = (datos_producto["Ganancia"] / datos_producto["Ingreso_total"]) * 100
                promedio_margen = datos_producto["Margen"].mean()
                
                margen_anual_promedio = datos_producto.groupby("Año")["Margen"].mean()
                variacion_margen_anual = margen_anual_promedio.pct_change().mean() * 100
                
                promedio_unidades = datos_producto["Unidades_vendidas"].mean()
                unidades_totales = datos_producto["Unidades_vendidas"].sum()
                
                unidades_por_año = datos_producto.groupby("Año")["Unidades_vendidas"].sum()
                variacion_unidades_anual = unidades_por_año.pct_change().mean() * 100
                
                col1, col2 = st.columns([0.25, 0.75])
                
                with col1:
                    st.metric(label="Precio Promedio", value=f"${promedio_precio:,.0f}".replace(",", "."), delta=f"{variacion_precio_anual:.2f}%")
                    st.metric(label="Margen Promedio", value=f"{promedio_margen:.0f}%".replace(",", "."), delta=f"{variacion_margen_anual:.2f}%")
                    st.metric(label="Unidades Vendidas", value=f"{unidades_totales:,.0f}".replace(",", "."), delta=f"{variacion_unidades_anual:.2f}%")
                
                with col2:
                    figura = graficar_tendencia_ventas(datos_producto, producto)
                    st.pyplot(figura)

if __name__ == "__main__":
    iniciar_aplicacion()