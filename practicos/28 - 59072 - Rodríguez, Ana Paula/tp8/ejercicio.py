import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuración inicial y tema
st.set_page_config(page_title="📊 Análisis de Datos de Productos", layout="wide")

# Función para mostrar información del alumno
def mostrar_informacion_alumno():
    with st.container():
        st.markdown("**Legajo1:** 59072")
        st.markdown("**Nombre Completo:** Rodriguez Ana Paula")
        st.markdown("**Comisión:** C7")

# Mostrar información del alumno solo si no hay datos cargados
st.sidebar.header("📤 Cargar datos")
archivo = st.sidebar.file_uploader("Subir archivo CSV", type="csv")

if archivo is None:
    mostrar_informacion_alumno()
else:
    try:
        # Leer datos del archivo CSV
        data = pd.read_csv(archivo)
        
        # Renombrar columnas para que coincidan con las esperadas
        data = data.rename(columns={
            "Unidades_vendidas": "CantidadVendida",
            "Ingreso_total": "ValorTotal",
            "Costo_total": "CostoTotal"
        })
        
        # Columnas requeridas después de renombrar
        columnas_requeridas = ["Sucursal", "Producto", "Año", "Mes", "CantidadVendida", "ValorTotal", "CostoTotal"]
        
        # Validación de columnas requeridas
        if all(col in data.columns for col in columnas_requeridas):
            # Conversión de columnas numéricas
            data["CantidadVendida"] = pd.to_numeric(data["CantidadVendida"], errors="coerce")
            data["ValorTotal"] = pd.to_numeric(data["ValorTotal"], errors="coerce")
            data["CostoTotal"] = pd.to_numeric(data["CostoTotal"], errors="coerce")
            
            # Cálculos adicionales
            data["PrecioUnitario"] = data["ValorTotal"] / data["CantidadVendida"]
            data["MargenBruto"] = (data["ValorTotal"] - data["CostoTotal"]) / data["ValorTotal"]
            
            # Resumen por producto
            resumen_producto = data.groupby("Producto").agg({
                "PrecioUnitario": "mean",
                "MargenBruto": "mean",
                "CantidadVendida": "sum"
            }).reset_index()
            
            # Filtro por sucursal
            sucursales = ["Todas"] + data["Sucursal"].unique().tolist()
            sucursal_seleccionada = st.sidebar.selectbox("🏢 Seleccionar sucursal", sucursales)

            if sucursal_seleccionada != "Todas":
                data = data[data["Sucursal"] == sucursal_seleccionada]
            
            # Encabezado principal
            st.header(f"📊 Datos de {'Todas las Sucursales' if sucursal_seleccionada == 'Todas' else sucursal_seleccionada}")
            
            # Mostrar datos por producto
            for _, fila in resumen_producto.iterrows():
                producto = fila["Producto"]
                precio_promedio = fila["PrecioUnitario"]
                margen_promedio = fila["MargenBruto"]
                unidades_vendidas = fila["CantidadVendida"]
                
                with st.container():
                    # Métricas y gráficos
                    columna_metrica, columna_grafico = st.columns([1, 3])
                    
                    # Métricas
                    with columna_metrica:
                        st.metric("💵 Precio Promedio", f"${precio_promedio:,.2f}")
                        st.metric("📈 Margen Promedio", f"{margen_promedio * 100:.2f}%")
                        st.metric("📦 Unidades Vendidas", f"{int(unidades_vendidas):,}")
                    
                    # Gráfico de evolución
                    with columna_grafico:
                        data_producto = data[data["Producto"] == producto]
                        data_producto["Fecha"] = pd.to_datetime(data_producto["Año"].astype(str) + "-" + data_producto["Mes"].astype(str) + "-01")
                        ventas_mensuales = data_producto.groupby("Fecha")["CantidadVendida"].sum().reset_index()

                        plt.figure(figsize=(8, 5))
                        plt.plot(ventas_mensuales["Fecha"], ventas_mensuales["CantidadVendida"], marker="o", color="#4A148C", label=producto)

                        # Línea de tendencia
                        z = np.polyfit(ventas_mensuales.index, ventas_mensuales["CantidadVendida"], 1)
                        p = np.poly1d(z)
                        plt.plot(ventas_mensuales["Fecha"], p(ventas_mensuales.index), "r--", label="Tendencia")
                        
                        plt.title(f"Evolución de Ventas - {producto}", fontsize=14, color="#4A148C")
                        plt.xlabel("Fecha")
                        plt.ylabel("Unidades Vendidas")
                        plt.grid(alpha=0.3)
                        plt.legend()
                        st.pyplot(plt)
        
        else:
            st.error("⚠️ El archivo no contiene las columnas requeridas.")
    
    except Exception as e:
        st.error(f"⚠️ Error al procesar el archivo: {e}")
