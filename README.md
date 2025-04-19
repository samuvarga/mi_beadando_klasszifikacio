# MI Beadandó Feladat

Ez a repository a Mesterséges Intelligencia tantárgy beadandó feladatának megoldását tartalmazza.

## Feladat leírása

A feladat két részből áll:
1.  **Klasszifikáció generált adathalmazon:** `make_blobs` segítségével generált adatokon három osztályozó (Naive Bayes, SVM, Logisztikus Regresszió) összehasonlítása. A kód a `part1_classification_generated.py` fájlban található.
2.  **Klasszifikáció valós adathalmazon:** A Wine adathalmaz betöltése és klasszifikálása egy választott algoritmussal (pl. k-NN). A kód a `part2_classification_wine.py` fájlban található.

## Futtatás

1.  **Klónozd a repository-t:**
    ```bash
    git clone <repository_url>
    cd mi2
    ```
2.  **Telepítsd a függőségeket:** (Érdemes virtuális környezetet használni)
    ```bash
    pip install numpy pandas scikit-learn matplotlib seaborn
    ```
    *(Opcionális: Hozz létre egy `requirements.txt` fájlt a `pip freeze > requirements.txt` paranccsal, és használd a `pip install -r requirements.txt` parancsot a telepítéshez.)*

3.  **Futtasd a szkripteket:**
    ```bash
    python part1_classification_generated.py
    python part2_classification_wine.py
    ```

## Eredmények

*(Itt összefoglalhatod a kapott pontosságokat és egyéb releváns metrikákat, esetleg beillesztheted a generált ábrákat is.)*