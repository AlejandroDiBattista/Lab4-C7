from fastapi import FastAPI, HTTPException, Depends
from sqlmodel import SQLModel, Field, Session, create_engine, select
from typing import List, Optional

## DEFINICION DEL MODELO DE DATOS

# Definir el modelo de datos
class Contacto(SQLModel, table=True): # table=True indica que se creará una tabla en la base de datos
    id: Optional[int] = Field(default=None, primary_key=True)   # primary_key indica que es la clave primaria
    nombre: str = Field(min_length=1, max_length=50)            # min_length y max_length son validaciones
    apellido: str 
    telefono: str

# Crear la base de datos y la sesión
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL, echo=True)

def initialize_db():
    # Crea las tablas en la base de datos
    SQLModel.metadata.create_all(engine)

    # Inserta datos iniciales en la base de datos
    with Session(engine) as session:
        # Trae un contacto de la base de datos
        contactos_existentes = session.exec(select(Contacto)).first()
        # Si no hay contactos en la base de datos, inserta algunos
        if not contactos_existentes:
            contactos_iniciales = [
                Contacto(nombre="Juan",  apellido="Perez",    telefono="123456789"),
                Contacto(nombre="Maria", apellido="Gomez",    telefono="987654321"),
                Contacto(nombre="Luis",  apellido="Martinez", telefono="555555555")
            ]
            session.add_all(contactos_iniciales) # Agrega todos los contactos a la sesión
            session.commit() # Confirma los cambios en la base de datos

initialize_db()


## DEFINICION DE LAS RUTAS DE LA API

# Crear la aplicación FastAPI
app = FastAPI()

# Dependencia para obtener la sesión de la base de datos
def get_session():
    with Session(engine) as session:
        yield session

# Ruta para crear un nuevo contacto
@app.post("/contactos/", response_model=Contacto)
def create_contacto(contacto: Contacto, session: Session = Depends(get_session)):
    session.add(contacto)
    session.commit()
    session.refresh(contacto)
    return contacto

# Ruta para obtener todos los contactos
@app.get("/contactos/", response_model=List[Contacto])
def read_contactos(skip: int = 0, limit: int = 10, session: Session = Depends(get_session)):
    comando = select(Contacto).offset(skip).limit(limit)
    contactos = session.exec(comando).all()
    return contactos

# Ruta para obtener un contacto por ID
@app.get("/contactos/{contacto_id}", response_model=Contacto)
def read_contacto(contacto_id: int, session: Session = Depends(get_session)):
    contacto = session.get(Contacto, contacto_id)
    if not contacto:
        raise HTTPException(status_code=404, detail="Contacto no encontrado")
    return contacto

# Ruta para actualizar un contacto
@app.put("/contactos/{contacto_id}", response_model=Contacto)
def update_contacto(contacto_id: int, contacto: Contacto, session: Session = Depends(get_session)):
    db_contacto = session.get(Contacto, contacto_id)
    if not db_contacto:
        raise HTTPException(status_code=404, detail="Contacto no encontrado")
    db_contacto.nombre = contacto.nombre
    db_contacto.apellido = contacto.apellido
    db_contacto.telefono = contacto.telefono
    session.commit()
    print(contacto.nombre_completo())
    session.refresh(db_contacto)
    return db_contacto

# Ruta para eliminar un contacto
@app.delete("/contactos/{contacto_id}", response_model=Contacto)
def delete_contacto(contacto_id: int, session: Session = Depends(get_session)):
    contacto = session.get(Contacto, contacto_id)
    if not contacto:
        raise HTTPException(status_code=404, detail="Contacto no encontrado")
    session.delete(contacto)
    session.commit()
    return contacto

# Inicializar la base de datos
initialize_db()


