Projekt inżynierski "Inteligentny licznik osób wykorzystujący głębokie sieci neuronowe"

Projekt stworzono z użyciem narzędzi w następujących wersjach:
    Anaconda 3 / Python 3.6
    TensorFlow 1.2.1
    OpenCV 3.3

W celu uruchomienia licznika należy w narzędziu Anaconda 3 utworzyć środowisko 'tensorflow-env' na podstawie pliku environment.yml.
conda env create -f environment.yml

Następnie, aby aktywować środowisko anacondy:
activate tensorflow-env
Aby włączyć program należy uruchomić plik Gui.py. Program może przetwarzać obraz z kamery, lub film z pliku, co należy ustawić w pliku Gui.py, linia 84.
	python Gui.py
Wyeksportowane raporty zapisują się do folderu reports.


Copyright © 2017 Katarzyna Gałka, Katarzyna Retowska, Zofia Rytlewska



