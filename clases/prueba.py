import streamlit as st
from PIL import Image
import exifread
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static

def obtener_coordenadas(exif_data):
    """Extrae las coordenadas GPS de los datos EXIF."""
    try:
        gps_latitude = exif_data.get('GPS GPSLatitude')
        gps_latitude_ref = exif_data.get('GPS GPSLatitudeRef')
        gps_longitude = exif_data.get('GPS GPSLongitude')
        gps_longitude_ref = exif_data.get('GPS GPSLongitudeRef')

        if not all([gps_latitude, gps_latitude_ref, gps_longitude, gps_longitude_ref]):
            return None

        def convertir_to_decimal(coord, ref):
            degrees, minutes, seconds = [float(x.num) / float(x.den) for x in coord.values]
            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
            if ref in ['S', 'W']:
                decimal = -decimal
            return decimal

        lat = convertir_to_decimal(gps_latitude, gps_latitude_ref.values)
        lon = convertir_to_decimal(gps_longitude, gps_longitude_ref.values)
        return lat, lon
    except Exception as e:
        return None

def main():
    st.title("Cargar Imagen y Mostrar Coordenadas en un Mapa")

    st.write("""
    Esta aplicación te permite cargar una imagen, extraer sus coordenadas GPS (si están disponibles) y mostrar la ubicación en un mapa interactivo.
    """)

    # Cargar imagen
    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Mostrar la imagen
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagen Cargada', use_column_width=True)

        # Extraer EXIF
        try:
            exif_data = exifread.process_file(uploaded_file, details=False)
            coordenadas = obtener_coordenadas(exif_data)

            if coordenadas:
                lat, lon = coordenadas
                st.success(f"**Coordenadas GPS encontradas:**\n- **Latitud:** {lat}\n- **Longitud:** {lon}")

                # Crear mapa
                m = folium.Map(location=[lat, lon], zoom_start=15)
                folium.Marker([lat, lon], popup="Ubicación de la Imagen").add_to(m)

                st.write("### Mapa de la Ubicación")
                folium_static(m)
            else:
                st.warning("No se encontraron coordenadas GPS en los datos EXIF de la imagen.")
        except Exception as e:
            st.error("Ocurrió un error al procesar la imagen.")

if __name__ == "__main__":
    main()
