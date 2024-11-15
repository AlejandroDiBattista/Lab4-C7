from fasthtml.common import *

app, rt = fast_app()

contador = 5

def Contador():
    global contador
    color = "green" if contador >= 0 else "red"
    return H5(f"Contador: {contador}", 
                id="contador",
                style=f"color:{color}"
              )

def Incrementar(cantidad):
    return A(f"{cantidad:+d}", 
             hx_put=f"/incrementar/{cantidad}", 
             hx_target="#contador"
             )

@rt('/')
def get():
    return Titled(
        "Mi Contador",
        Incrementar(-1)," | ",Incrementar(-5)," | ",Incrementar(-10),
        Contador(),
        Incrementar(1)," | ",Incrementar(5)," | ",Incrementar(10)
    )

@rt('/incrementar/{cantidad}')
def put(cantidad:int):
    global contador
    contador +=  cantidad
    return  Contador()

serve()