def suma(a:int,b:int):
    """
        La funcion suma toma
        2 parametros 
        y retorna la suma
    """
    return a+b

print(suma(10,20))
print(suma("Hola ", "Mundo"))

print(suma.__annotations__ )
print(suma.__doc__)


# class Contacto:
#     def __init__(nombre, apellido, telefono):
#         self.nombre = nombre
#         self.apellido = apellido
#         self.telefono = telefono

# c = Contacto(nombre="ALej", apellido="Dib", telefono="333")
# c.nombre = "Alejandro"

# a = open("telefono.txt")
# lista = a.readlines()
# a.close()

# with open("telefono.txt") as a:
#     lista = a.readlines()

# # streamlit 

# st.header("Hola")
# st.sidebar.header("Hello")
# st.sidebar.write("Esto es un texto")

# with st.sidebar:
#     st.header("Hello")
#     st.write("Esto es un texto")

import time
import contextlib

nivel = 0
@contextlib.contextmanager
def medir():
    global nivel
    nivel += 1
    print(" " * nivel, "Inicio")
    inicio = time.time()
    yield
    fin = time.time()
    print(" " * nivel, f"Tiempo: {fin - inicio}")
    nivel -= 1

with medir():
    time.sleep(1)
    with medir():
        time.sleep(2)
