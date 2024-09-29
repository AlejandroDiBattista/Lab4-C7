def asignar_puntaje(carta_numero, carta_palo):
    tabla_valores = {
        (1, 'espada'): 14,
        (1, 'basto'): 13,
        (7, 'oro'): 12,
        (7, 'copa'): 11,
        3: 10,
        2: 9,
        1: 8,
        12: 7,
        11: 6,
        10: 5,
        7: 4,
        6: 3,
        5: 2,
        4: 1
    }
    return tabla_valores.get((carta_numero, carta_palo), tabla_valores.get(carta_numero, 0))

palos_visuales = {'oro': 'oro', 'copa': 'copa', 'espada': 'espada', 'basto': 'basto'}


class Baraja:
    def _init_(self, carta_numero, carta_palo):
        self.carta_numero = carta_numero
        self.carta_palo = carta_palo

    @property
    def valor(self):
        return asignar_puntaje(self.carta_numero, self.carta_palo)

    def _str_(self):
        return f"{self.carta_numero:2}{palos_visuales[self.carta_palo]}"

    def _repr_(self):
        return self._str_()

    def _gt_(self, otra_carta):
        return self.valor > otra_carta.valor


carta_ejemplo = Baraja(1, "oro")
print(carta_ejemplo.carta_numero)  # 1
print(carta_ejemplo.carta_palo)  # oro
print(carta_ejemplo.valor)  # 8
print(str(carta_ejemplo))  # 1 de oro
print(repr(carta_ejemplo))
carta_1 = Baraja(7, "copa")
carta_2 = Baraja(7, "oro")
print(carta_1, carta_2)  # False

import random

lista_numeros = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
lista_palos = ['oro', 'copa', 'espada', 'basto']


class Mazzo:
    def _init_(self):
        self.todas_cartas = [Baraja(carta_numero, carta_palo) for carta_numero in lista_numeros for carta_palo in lista_palos]
        self.mezclar()

    def mezclar(self):
        random.shuffle(self.todas_cartas)

    def repartir(self, cantidad):
        return [self.todas_cartas.pop() for _ in range(cantidad)]


mazzo_ejemplo = Mazzo()
mazzo_ejemplo.mezclar()
print(mazzo_ejemplo.todas_cartas)
print(mazzo_ejemplo.repartir(3))


class Participante:
    def _init_(self, nombre_participante):
        self.nombre_participante = nombre_participante
        self.cartas_en_mano = []
        self.puntos_totales = 0

    def recibir_cartas(self, cartas):
        self.cartas_en_mano = cartas

    def lanzar_una_carta(self):
        random.shuffle(self.cartas_en_mano)
        return self.cartas_en_mano.pop()

    def _str_(self):
        return self.nombre_participante

    def _repr_(self):
        return self._str_()


jugador_prueba = Participante("Luis")
jugador_prueba.recibir_cartas(mazzo_ejemplo.repartir(3))

print(jugador_prueba.cartas_en_mano)
print(jugador_prueba.lanzar_una_carta())
print(jugador_prueba.cartas_en_mano)


class Juego:
    def _init_(self, participante_1, participante_2):
        self.participantes = [participante_1, participante_2]
        self.mazzo = Mazzo()

    def jugar_ronda(self):
        return [[jugador.lanzar_una_carta() for jugador in self.participantes] for _ in range(3)]

    def evaluar_ronda(self, jugadas):
        rondas_ganadas = [sum(1 for carta1, carta2 in jugadas if carta1 > carta2),
                          sum(1 for carta1, carta2 in jugadas if carta2 > carta1)]

        for victorias, participante in zip(rondas_ganadas, self.participantes):
            if victorias >= 2:
                participante.puntos_totales += 1
                print(f"{participante} gana la ronda")

    def mostrar_puntajes(self):
        print(f"\nPuntajes:")
        for participante in self.participantes:
            print(f" - {participante}: {participante.puntos_totales}")
        print('-' * 20)

    def iniciar_juego(self):
        ronda_numero = 0
        while all(jugador.puntos_totales < 15 for jugador in self.participantes):
            self.mazzo = Mazzo()
            ronda_numero += 1
            print(f"Ronda {ronda_numero} {[jugador for jugador in self.participantes]}")

            for jugador in self.participantes:
                jugador.recibir_cartas(self.mazzo.repartir(3))

            jugadas = self.jugar_ronda()
            for jugada in jugadas:
                print(' ', jugada)
            self.evaluar_ronda(jugadas)

            self.mostrar_puntajes()

        for jugador in self.participantes:
            if jugador.puntos_totales >= 15:
                print(f"{jugador} gana el juego")


partida = Juego(Participante("Mario"), Participante("Laura"))
partida.iniciar_juego()