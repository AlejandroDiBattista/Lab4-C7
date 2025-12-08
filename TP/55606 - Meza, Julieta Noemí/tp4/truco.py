def obtener_puntaje(valor_figura, tipo_mazo):
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
    return tabla_valores.get((valor_figura, tipo_mazo), tabla_valores.get(valor_figura, 0))

descripcion_mazo = {'oro': 'oro', 'copa': 'copa', 'espada': 'espada', 'basto': 'basto'}


class BarajaCarta:
    def _init_(self, valor_figura, tipo_mazo):
        self.valor_figura = valor_figura
        self.tipo_mazo = tipo_mazo

    @property
    def puntaje(self):
        return obtener_puntaje(self.valor_figura, self.tipo_mazo)

    def _str_(self):
        return f"{self.valor_figura:2}{descripcion_mazo[self.tipo_mazo]}"

    def _repr_(self):
        return self._str_()

    def _gt_(self, otro_carta):
        return self.puntaje > otro_carta.puntaje


carta_test = BarajaCarta(1, "oro")
print(carta_test.valor_figura)  # 1
print(carta_test.tipo_mazo)  # oro
print(carta_test.puntaje)  # 8
print(str(carta_test))  # 1 de oro
print(repr(carta_test))
carta_a = BarajaCarta(7, "copa")
carta_b = BarajaCarta(7, "oro")
print(carta_a, carta_b)  # False

import random

lista_figuras = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
lista_mazos = ['oro', 'copa', 'espada', 'basto']


class JuegoBaraja:
    def _init_(self):
        self.naipes = [BarajaCarta(valor_figura, tipo_mazo) for valor_figura in lista_figuras for tipo_mazo in lista_mazos]
        self.barajar()

    def barajar(self):
        random.shuffle(self.naipes)

    def repartir_naipes(self, cantidad):
        return [self.naipes.pop() for _ in range(cantidad)]


baraja_test = JuegoBaraja()
baraja_test.barajar()
print(baraja_test.naipes)
print(baraja_test.repartir_naipes(3))


class JugadorMesa:
    def _init_(self, alias):
        self.alias = alias
        self.mano = []
        self.puntaje_total = 0

    def recibir_naipes(self, naipes_recibidos):
        self.mano = naipes_recibidos

    def lanzar_naipe(self):
        random.shuffle(self.mano)
        return self.mano.pop()

    def _str_(self):
        return self.alias

    def _repr_(self):
        return self._str_()


jugador_prueba = JugadorMesa("Pedro")
jugador_prueba.recibir_naipes(baraja_test.repartir_naipes(3))

print(jugador_prueba.mano)
print(jugador_prueba.lanzar_naipe())
print(jugador_prueba.mano)


class JuegoMesa:
    def _init_(self, jugador_a, jugador_b):
        self.lista_jugadores = [jugador_a, jugador_b]
        self.baraja = JuegoBaraja()

    def ronda_completa(self):
        return [[jugador.lanzar_naipe() for jugador in self.lista_jugadores] for _ in range(3)]

    def evaluar_ronda(self, jugadas):
        resultado_ronda = [sum(1 for carta_a, carta_b in jugadas if carta_a > carta_b),
                           sum(1 for carta_a, carta_b in jugadas if carta_b > carta_a)]

        for puntos_ronda, jugador in zip(resultado_ronda, self.lista_jugadores):
            if puntos_ronda >= 2:
                jugador.puntaje_total += 1
                print(f"{jugador} gana la ronda")

    def mostrar_puntajes(self):
        print(f"\nPuntajes:")
        for jugador in self.lista_jugadores:
            print(f" - {jugador}: {jugador.puntaje_total}")
        print('-' * 20)

    def iniciar_juego(self):
        numero_ronda = 0
        while all(jugador.puntaje_total < 15 for jugador in self.lista_jugadores):
            self.baraja = JuegoBaraja()
            numero_ronda += 1
            print(f"Ronda {numero_ronda} {[jugador for jugador in self.lista_jugadores]}")

            for jugador in self.lista_jugadores:
                jugador.recibir_naipes(self.baraja.repartir_naipes(3))

            jugadas_ronda = self.ronda_completa()
            for jugada in jugadas_ronda:
                print(' ', jugada)
            self.evaluar_ronda(jugadas_ronda)

            self.mostrar_puntajes()

        for jugador in self.lista_jugadores:
            if jugador.puntaje_total >= 15:
                print(f"{jugador} gana la partida")


partida_test = JuegoMesa(JugadorMesa("Ricardo"), JugadorMesa("Julieta"))
partida_test.iniciar_juego()