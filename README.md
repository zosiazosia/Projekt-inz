# zmirlacz
Projekt inżynierski "Inteligentny licznik osób"

Aktualnie przyjęty kierunek wejścia - w lewo. W celu zmiany kierunku wejścia należy przy tworzeniu obiektu counter podać w nawiasie 'right'
Przyjęto założenie, że pomieszczenie przed rozpoczęciem zliczania jest puste

Aktualnie nie jest używana funkcja counter.come_out -> przy wyjściu ZAWSZE następuje reidentyfikacja

Do testowania innych rodzajów klasyfikacji ( wszystko w pliku Transform.py):
- zmiana sieci: self.base_model
- zmiana warstwy: self. model zmienić string w base_model.get_layer
- zmiana transformacji wektorów: funkcja transform  <- TO TRZEBA NA PEWNO ZROBIĆ
    'a' zostaje to samo, to jest wyjście z sieci, aktualnie o kształcie 1x14x14x512
    trzeba pobawić się z jego rozdziałem na mniejsze kawałki
    polecam testować zachowanie tablic na małych przykładach gdzieś osobno
- zmiana samej klasyfikacji: w funkcjach tree_decideIn i tree_decideOut
    to co podajemy w nawiasie query powinno być może jakąś średnią wszystkich wektorów danej postaci a nie tylko pierwszym z nich
    ewentualnie (może to byłoby nawet lepsze rozwiązanie) można robić takie zapytanie dla wszystkich wektorów
   danej postaci po kolei i na tej podstawie decydować. Samo zapytanie do drzewa nie trwa raczej zbyt długo