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
- zmiana samej klasyfikacji: w funkcjach tree_decideIn i tree_decideOut
    to co podajemy w nawiasie query powinno być może jakąś średnią wszystkich wektorów danej postaci a nie tylko pierwszym z nich
    ewentualnie (może to byłoby nawet lepsze rozwiązanie) można robić takie zapytanie dla wszystkich wektorów
   danej postaci po kolei i na tej podstawie decydować. Samo zapytanie do drzewa nie trwa raczej zbyt długo