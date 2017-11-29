# zmirlacz
Projekt inżynierski "Inteligentny licznik osób"

Aktualnie przyjęty kierunek wejścia - w lewo. W celu zmiany kierunku wejścia należy przy tworzeniu obiektu counter podać w nawiasie 'right'
Przyjęto założenie, że pomieszczenie przed rozpoczęciem zliczania jest puste

Aktualnie nie jest używana funkcja counter.come_out -> przy wyjściu ZAWSZE następuje reidentyfikacja

Do testowania innych rodzajów klasyfikacji ( wszystko w pliku Transform.py):
- zmiana sieci: self.base_model
- zmiana warstwy: self. model zmienić string w base_model.get_layer
- zmiana transformacji wektorów: funkcja transform
    'a' zostaje to samo, to jest wyjście z sieci, aktualnie o kształcie 1x14x14x512
    ZOFIA EDITS: dodałam dwie możliwości w funkcji transform. pierwsza rozkłada na 3 części drugi wymiar wektora 'x' (0:4:10:14),
    druga rozkłada trzeci wymiar na 3 części (0:4:10:14).
    Mamy teraz 3 możliwości transformacji, zostaje przetestować która daje najlepsze wyniki
- zmiana samej klasyfikacji: w funkcji tree_decide:
    możliwe do wyboru funkcje (jako return):
        - self.mostFreqNearest(vectors, tree, indexes, direction) -> zwraca osobę, która najczęściej jest pierwsza na liście
        - self.kMultiplyDistance(vectors, tree, indexes, direction, 5) -> zwraca osobę, która ma najmniejszą średnią odległość -> brane pod uwagę pierwsze k wektorów dla każdego z 10 zapisanych wektorów postaci

