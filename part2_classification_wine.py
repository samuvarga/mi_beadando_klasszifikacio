import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier # Példaként k-NN-t használunk
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Adat betöltése ---
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
# Az oszlopnevek a wine.names fájlból vagy a dataset leírásából származnak
column_names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
                'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
                'Proanthocyanins', 'Color intensity', 'Hue',
                'OD280/OD315 of diluted wines', 'Proline']

try:
    df = pd.read_csv(url, header=None, names=column_names)
    print("Wine adathalmaz sikeresen betöltve.")
    print(f"Adathalmaz mérete: {df.shape}")
    print("Adathalmaz első 5 sora:")
    print(df.head())
    print("\nInformáció az adathalmazról:")
    df.info() # Ellenőrizzük nincsenek-e hiányzó értékek
    print("\nOsztályok eloszlása:")
    print(df['Class'].value_counts())

    # --- Adat előkészítése ---
    X = df.drop('Class', axis=1) # Jellemzők
    y = df['Class']             # Célváltozó (osztályok)

    # Adatok felosztása tanító és teszt halmazra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) # Stratify a kiegyensúlyozott felosztáshoz

    # Jellemzők skálázása (fontos lehet pl. k-NN, SVM, LogReg esetén)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Modell tanítása és kiértékelése (pl. k-NN) ---
    model_name = "k-Nearest Neighbors (k-NN)"
    knn = KNeighborsClassifier(n_neighbors=5) # k=5 egy gyakori választás

    print(f"\n--- {model_name} ---")
    # Tanítás skálázott adatokon
    knn.fit(X_train_scaled, y_train)

    # Jóslás skálázott teszt adatokon
    y_pred = knn.predict(X_test_scaled)

    # Kiértékelés
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Pontosság (Accuracy): {accuracy:.4f}")
    print("Osztályozási riport:")
    print(report)
    print("Konfúziós mátrix:")

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=knn.classes_, yticklabels=knn.classes_)
    plt.xlabel('Jósolt osztály')
    plt.ylabel('Valós osztály')
    plt.title(f'{model_name} - Konfúziós Mátrix')
    plt.show()

except Exception as e:
    print(f"Hiba történt az adatok letöltése vagy feldolgozása közben: {e}")
    print("Ellenőrizd az internetkapcsolatot és a megadott URL-t.")
    print("Alternatívaként töltsd le a 'wine.data' fájlt manuálisan és helyezd a projekt mappájába, majd módosítsd a betöltési részt.")
    # Példa lokális fájl betöltésére:
    # filepath = 'wine.data'
    # try:
    #     df = pd.read_csv(filepath, header=None, names=column_names)
    #     # ... a kód többi része ...
    # except FileNotFoundError:
    #     print(f"Hiba: A '{filepath}' fájl nem található.")
