import pandas as pd
import time # Időméréshez
import os   # Mappa létrehozásához
from sklearn.model_selection import train_test_split, GridSearchCV # GridSearchCV hozzáadva
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # PCA hozzáadva
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB # Naive Bayes hozzáadva
from sklearn.svm import SVC # SVM hozzáadva
from sklearn.linear_model import LogisticRegression # LogReg hozzáadva
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Numpy importálása (PCA legendához kellhet)

# --- Mappa létrehozása az ábráknak ---
output_dir_part2 = "part2_plots"
if not os.path.exists(output_dir_part2):
    os.makedirs(output_dir_part2)
    print(f"Létrehozva a '{output_dir_part2}' mappa az ábrák mentéséhez.")

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

    # --- Adat vizualizálásához (PCA-val 2D-ben) ---
    # Jellemzők és célváltozó szétválasztása a vizualizációhoz
    X_viz = df.drop('Class', axis=1)
    y_viz = df['Class']

    # Skálázás PCA előtt
    scaler_pca = StandardScaler()
    X_viz_scaled = scaler_pca.fit_transform(X_viz)

    pca = PCA(n_components=2, random_state=42) # random_state a reprodukálhatósághoz
    X_pca = pca.fit_transform(X_viz_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_viz, cmap='viridis', edgecolor='k', s=50)
    plt.title('Wine adathalmaz vizualizációja (PCA után)')
    plt.xlabel('Első főkomponens')
    plt.ylabel('Második főkomponens')
    # Legenda létrehozása (feltételezve, hogy az osztályok 1, 2, 3)
    handles, labels = scatter.legend_elements()
    class_labels = [f'Osztály {i}' for i in sorted(y_viz.unique())] # Dinamikus címkék
    plt.legend(handles=handles, labels=class_labels)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir_part2, 'wine_data_pca.png')) # Mentés
    plt.show()


    # --- Adat előkészítése ---
    X = df.drop('Class', axis=1) # Jellemzők
    y = df['Class']             # Célváltozó (osztályok)

    # Adatok felosztása tanító és teszt halmazra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) # Stratify a kiegyensúlyozott felosztáshoz

    # Jellemzők skálázása (fontos lehet pl. k-NN, SVM, LogReg esetén)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Modellek definiálása ---
    models = {
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(random_state=42, probability=True), # probability=True lehet hasznos később
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=10000), # Növelt max_iter a konvergenciához
        "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5) # k-NN-t ide is tehetjük - KOMMENT ELTÁVOLÍTVA
    }

    # --- Modellek tanítása és kiértékelése ---
    results_part2 = {}
    training_times_part2 = {}

    print("\n--- Modellek Kiértékelése ---")

    for name, model in models.items():
        print(f"\n--- {name} ---")
        start_time = time.time()

        # Tanítás skálázott adatokon
        model.fit(X_train_scaled, y_train)
        end_time = time.time()
        training_times_part2[name] = end_time - start_time

        # Jóslás (tanító és teszt)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        # Kiértékelés
        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        report_test = classification_report(y_test, y_pred_test)
        cm_test = confusion_matrix(y_test, y_pred_test)

        results_part2[name] = {
            "model": model,
            "accuracy_train": accuracy_train,
            "accuracy_test": accuracy_test,
            "report_test": report_test,
            "confusion_matrix_test": cm_test
        }

        print(f"Tanítási idő: {training_times_part2[name]:.4f} másodperc")
        print(f"Pontosság (Tanító halmaz): {accuracy_train:.4f}")
        print(f"Pontosság (Teszt halmaz): {accuracy_test:.4f}")

        # --- Tanult paraméterek kiírása (opcionális, mint Part 1-ben) ---
        if isinstance(model, LogisticRegression):
            print(f"  LogReg Együtthatók (coef_) alakja: {model.coef_.shape}")
            print(f"  LogReg Intercept (intercept_) alakja: {model.intercept_.shape}")
        elif isinstance(model, SVC):
            # Csak akkor írjuk ki, ha a kernel lineáris, vagy ha a támaszvektorok száma érdekes
            print(f"  SVM Támaszvektorok száma osztályonként (n_support_): {model.n_support_}")
        elif isinstance(model, GaussianNB):
            print("  Naive Bayes modell tanítva.")
        # --- Paraméterek kiírása vége ---

        print("Osztályozási riport (Teszt halmaz):")
        print(report_test)
        print("Konfúziós mátrix (Teszt halmaz):")

        # Plot confusion matrix
        plt.figure(figsize=(6, 4))
        # A címkék legyenek az osztályok (1, 2, 3)
        class_labels_sorted = sorted(y.unique())
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_labels_sorted, yticklabels=class_labels_sorted)
        plt.xlabel('Jósolt osztály')
        plt.ylabel('Valós osztály')
        plt.title(f'{name} - Konfúziós Mátrix (Teszt)')
        # Fájlnév létrehozása
        safe_name = name.replace(" ", "_").replace("-", "").replace("(", "").replace(")", "").lower()
        plt.savefig(os.path.join(output_dir_part2, f'confusion_matrix_{safe_name}.png')) # Mentés
        plt.show()

    # --- Eredmények összefoglalása ---
    print("\n--- Összehasonlítás ---")
    print(f"{'Modell':<25} {'Tanítási idő (s)':<20} {'Tanuló pontosság':<20} {'Teszt pontosság':<20}")
    print("-" * 85)
    table_data_part2 = []
    model_names_part2 = list(results_part2.keys())
    for name in model_names_part2:
        metrics = results_part2[name]
        print(f"{name:<25} {training_times_part2[name]:<20.4f} {metrics['accuracy_train']:<20.4f} {metrics['accuracy_test']:<20.4f}")
        # Adatok gyűjtése a táblázathoz
        table_data_part2.append([f"{training_times_part2[name]:.4f}",
                                 f"{metrics['accuracy_train']:.4f}",
                                 f"{metrics['accuracy_test']:.4f}"])

    # --- Táblázat készítése az összefoglaló eredményekről ---
    fig_table, ax_table = plt.subplots(figsize=(10, max(2, len(model_names_part2) * 0.5))) # Dinamikus magasság
    ax_table.axis('tight')
    ax_table.axis('off')
    columns_part2 = ['Tanítási idő (s)', 'Tanuló pontosság', 'Teszt pontosság']
    the_table_part2 = ax_table.table(cellText=table_data_part2,
                                     rowLabels=model_names_part2,
                                     colLabels=columns_part2,
                                     loc='center',
                                     cellLoc='center')

    the_table_part2.auto_set_font_size(False)
    the_table_part2.set_fontsize(10)
    the_table_part2.scale(1.2, 1.2) # Méretarány növelése

    plt.title('Modellek Összehasonlító Eredményei (Wine Adathalmaz)', y=1.1, fontsize=12) # Cím hozzáadása
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_part2, 'summary_results_table_part2.png'), bbox_inches='tight') # Mentés
    print(f"\nÖsszefoglaló táblázat mentve: {os.path.join(output_dir_part2, 'summary_results_table_part2.png')}")
    plt.show()

    # --- Tanulási/Tesztelési Pontosság az Iterációk Függvényében (Logisztikus Regresszió) ---
    print("\n--- Tanulási/Tesztelési Pontosság az Iterációk Függvényében (Logisztikus Regresszió) ---")

    iterations_lr_part2 = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000] # Kipróbált iterációszámok
    train_accuracies_lr_part2 = []
    test_accuracies_lr_part2 = []

    # Figyelmeztetések ideiglenes kikapcsolása (ConvergenceWarning miatt)
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for n_iter in iterations_lr_part2:
        # Modell létrehozása az adott iterációszámmal
        # Használjuk ugyanazt a random_state-et, mint a fő modellnél
        lr_iter_part2 = LogisticRegression(random_state=42, max_iter=n_iter, solver='lbfgs')
        # Tanítás
        lr_iter_part2.fit(X_train_scaled, y_train)
        # Pontosság számítása
        train_acc = lr_iter_part2.score(X_train_scaled, y_train)
        test_acc = lr_iter_part2.score(X_test_scaled, y_test)
        train_accuracies_lr_part2.append(train_acc)
        test_accuracies_lr_part2.append(test_acc)
        # print(f"Iterációk: {n_iter}, Tanuló pontosság: {train_acc:.4f}, Teszt pontosság: {test_acc:.4f}") # Opcionális

    # Figyelmeztetések visszaállítása
    warnings.filterwarnings("default", category=ConvergenceWarning)

    # Ábrázolás
    plt.figure(figsize=(10, 6))
    plt.plot(iterations_lr_part2, train_accuracies_lr_part2, marker='o', label='Tanuló pontosság')
    plt.plot(iterations_lr_part2, test_accuracies_lr_part2, marker='x', linestyle='--', label='Teszt pontosság')
    plt.xlabel("Iterációk száma (max_iter)")
    plt.ylabel("Pontosság")
    plt.title("Logisztikus Regresszió Pontossága az Iterációk Függvényében (Wine)")
    plt.xscale('log') # Logaritmikus skála az x tengelyen
    plt.ylim(0.9, 1.02) # y tengely limitálása a jobb láthatóságért
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(os.path.join(output_dir_part2, 'logistic_regression_accuracy_vs_iterations_part2.png')) # Mentés
    print(f"LogReg iterációs ábra mentve: {os.path.join(output_dir_part2, 'logistic_regression_accuracy_vs_iterations_part2.png')}")
    plt.show()


    # --- Hiperparaméter-hangolás (Példa: SVM 'C' és 'gamma' paraméterei) ---
    print("\n--- Hiperparaméter-hangolás (SVM) ---")

    # Paraméter rács definiálása
    param_grid_svm = {
        'C': [0.1, 1, 10, 100],          # Regularizációs paraméter
        'gamma': ['scale', 'auto', 0.1, 1], # Kernel együttható (RBF kernelhez)
        'kernel': ['rbf']                # Csak RBF kernelt próbálunk (gyakori választás)
    }

    # SVC modell létrehozása (nem kell random_state a GridSearchCV-hez)
    svm_cv = SVC(probability=True) # probability=True, ha később kell a legjobb modellnek

    # GridSearchCV létrehozása
    # cv=5 -> 5-szörös keresztvalidáció
    # scoring='accuracy' -> pontosság alapján választjuk a legjobb modellt
    # n_jobs=-1 -> használja az összes elérhető CPU magot a gyorsításhoz
    grid_search_svm = GridSearchCV(estimator=svm_cv,
                                   param_grid=param_grid_svm,
                                   cv=5,
                                   scoring='accuracy',
                                   n_jobs=-1,
                                   verbose=1) # Verbose=1 mutatja a folyamatot

    print("GridSearchCV indítása SVM-re...")
    start_time_gs = time.time()
    # Illesztés a skálázott tanító adatokra
    grid_search_svm.fit(X_train_scaled, y_train)
    end_time_gs = time.time()
    print(f"GridSearchCV befejezve. Idő: {end_time_gs - start_time_gs:.2f} másodperc")

    # Legjobb paraméterek és pontosság kiírása
    print(f"\nLegjobb paraméterek (SVM): {grid_search_svm.best_params_}")
    print(f"Legjobb keresztvalidációs pontosság (SVM): {grid_search_svm.best_score_:.4f}")

    # Értékeljük ki a legjobb (hangolt) SVM modellt a teszt adatokon
    best_svm_part2 = grid_search_svm.best_estimator_
    y_pred_svm_best_part2 = best_svm_part2.predict(X_test_scaled)
    accuracy_svm_best_test_part2 = accuracy_score(y_test, y_pred_svm_best_part2)

    print(f"\nLegjobb (hangolt) SVM modell pontossága a teszt halmazon: {accuracy_svm_best_test_part2:.4f}")
    # --- Legjobb SVM paraméterek kiírása ---
    print(f"  Legjobb SVM Támaszvektorok száma osztályonként (n_support_): {best_svm_part2.n_support_}")
    # --- Paraméterek kiírása vége ---
    print("Osztályozási riport (Legjobb SVM, Teszt halmaz):")
    print(classification_report(y_test, y_pred_svm_best_part2))

    # Konfúziós mátrix a legjobb SVM modellhez
    cm_svm_best = confusion_matrix(y_test, y_pred_svm_best_part2)
    plt.figure(figsize=(6, 4))
    class_labels_sorted = sorted(y.unique()) # Újra lekérjük, biztos ami biztos
    sns.heatmap(cm_svm_best, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels_sorted, yticklabels=class_labels_sorted)
    plt.xlabel('Jósolt osztály')
    plt.ylabel('Valós osztály')
    plt.title('Legjobb (hangolt) SVM - Konfúziós Mátrix (Teszt)')
    plt.savefig(os.path.join(output_dir_part2, 'confusion_matrix_svm_best_tuned.png')) # Mentés
    print(f"Legjobb SVM konfúziós mátrixa mentve: {os.path.join(output_dir_part2, 'confusion_matrix_svm_best_tuned.png')}")
    plt.show()

    # --- Hiperparaméter-hangolás Eredményeinek Vizualizációja (SVM Heatmap) ---
    print("\n--- SVM Hiperparaméter-hangolás Vizualizációja ---")

    # Eredmények kinyerése a GridSearchCV-ből
    results_df = pd.DataFrame(grid_search_svm.cv_results_)

    # Csak a releváns oszlopok kiválasztása és típuskonverzió, ha szükséges
    # A paraméter nevek a param_grid kulcsaiból származnak ('param_' előtaggal)
    results_df = results_df[['param_C', 'param_gamma', 'mean_test_score']]
    # A 'gamma' lehet string ('scale', 'auto') vagy float. A heatmaphez numerikus értékek kellenek.
    # Egyszerűsítésként most csak a numerikus gamma értékekkel dolgozunk, ha vannak.
    # Vagy átalakíthatnánk a stringeket is valamilyen numerikus reprezentációra, de az bonyolultabb.
    # Itt most feltételezzük, hogy a param_grid csak numerikus gamma értékeket is tartalmazott,
    # vagy a DataFrame képes kezelni a kevert típusokat a pivotáláshoz.
    # A biztonság kedvéért konvertáljuk a pontszámot float-ra.
    results_df['mean_test_score'] = results_df['mean_test_score'].astype(float)

    # Pivot tábla létrehozása a heatmaphez
    # A 'gamma' oszlop tartalmazhat stringeket ('scale', 'auto'), ezeket kezelni kell.
    # Legegyszerűbb, ha a pivotálás előtt ezeket kiszűrjük vagy átalakítjuk.
    # Most próbáljuk meg közvetlenül, hátha a pandas kezeli.
    try:
        scores = results_df.pivot(index='param_C', columns='param_gamma', values='mean_test_score')

        plt.figure(figsize=(10, 6))
        sns.heatmap(scores, annot=True, fmt=".4f", cmap="viridis") # fmt=".4f" a pontosság jobb megjelenítéséhez
        plt.title('SVM Keresztvalidációs Pontosság (Heatmap)')
        plt.xlabel('Gamma paraméter')
        plt.ylabel('C paraméter')
        plt.savefig(os.path.join(output_dir_part2, 'svm_tuning_heatmap.png')) # Mentés
        print(f"SVM hangolás heatmap mentve: {os.path.join(output_dir_part2, 'svm_tuning_heatmap.png')}")
        plt.show()

    except ValueError as ve:
        print(f"\nHiba a heatmap készítésekor: {ve}")
        print("Lehetséges ok: A 'gamma' paraméter tartalmazott nem numerikus értékeket ('scale', 'auto'), amelyeket a pivot nem tudott kezelni.")
        print("A heatmap nem készült el. Próbáld meg csak numerikus 'gamma' értékekkel a param_grid-ben.")
    except KeyError as ke:
         print(f"\nHiba a heatmap készítésekor: Hiányzó oszlop - {ke}")
         print("Ellenőrizd a DataFrame oszlopneveit és a pivot függvény paramétereit.")


    # --- Döntési határok vizualizációja (2D PCA adatokon) ---
    print("\n--- Döntési határok vizualizációja (PCA adatokon) ---")

    # PCA adatok felosztása tanító/teszt halmazra (ugyanazzal a random_state-tel)
    # X_pca már létezik a korábbi vizualizációból
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        X_pca, y, test_size=0.3, random_state=42, stratify=y
    )

    # Modellek definiálása a PCA adatokhoz (használjuk a legjobb SVM-et)
    # Újra létrehozzuk őket, hogy biztosan a PCA adatokon legyenek tanítva
    models_pca = {
        "Naive Bayes": GaussianNB(),
        # A legjobb, hangolt SVM modellt használjuk itt is
        "Best Tuned SVM": grid_search_svm.best_estimator_,
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=10000)
    }

    # Meshgrid létrehozása
    h = .02  # Lépésköz a mesh-ben
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Ábra létrehozása subplotokhoz
    n_models = len(models_pca)
    fig_db, axes_db = plt.subplots(1, n_models, figsize=(n_models * 5, 5)) # Méret igazítása
    # Ha csak egy modell van, az axes_db nem tömb, ezt kezelni kell
    if n_models == 1:
        axes_db = [axes_db]

    plot_idx = 0
    all_handles_pca = []
    all_labels_pca = []
    class_labels_pca = [f'Osztály {i}' for i in sorted(y.unique())] # Címkék a legendához

    for name, model_pca in models_pca.items():
        print(f"Döntési határok rajzolása: {name}")
        # Modell tanítása a 2D PCA adatokon
        model_pca.fit(X_train_pca, y_train_pca)

        # Jóslás a meshgrid pontjaira
        Z = model_pca.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Subplot kiválasztása
        ax_db = axes_db[plot_idx]

        # Döntési régiók rajzolása
        ax_db.contourf(xx, yy, Z, cmap=plt.cm.viridis, alpha=0.6)

        # Tanító pontok rajzolása
        scatter_train_pca = ax_db.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_pca,
                                          cmap=plt.cm.viridis, edgecolor='k', s=20, label='Tanító')
        # Teszt pontok rajzolása (más markerrel)
        scatter_test_pca = ax_db.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test_pca,
                                         cmap=plt.cm.viridis, edgecolor='k', marker='^', s=30, label='Teszt')

        ax_db.set_xlim(xx.min(), xx.max())
        ax_db.set_ylim(yy.min(), yy.max())
        ax_db.set_xticks(())
        ax_db.set_yticks(())
        ax_db.set_title(name)

        # Legenda elemek gyűjtése az első subplotból
        if plot_idx == 0:
            handles_train, _ = scatter_train_pca.legend_elements()
            # Csak egy handle kell a teszt pontokhoz
            handles_test = [scatter_test_pca]
            all_handles_pca.extend(handles_train)
            all_handles_pca.extend(handles_test)
            # Címkék: Osztály 1, Osztály 2, Osztály 3, Teszt
            all_labels_pca.extend(class_labels_pca)
            all_labels_pca.append('Teszt pontok')


        plot_idx += 1

    # Közös legenda hozzáadása
    fig_db.legend(all_handles_pca, all_labels_pca, loc='lower center', ncol=len(all_labels_pca), bbox_to_anchor=(0.5, -0.05))
    fig_db.suptitle('Modellek Döntési Határai (PCA-redukált Wine Adatokon)', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Hely hagyása a suptitle-nek és legendának
    plt.savefig(os.path.join(output_dir_part2, 'decision_boundaries_pca_part2.png'), bbox_inches='tight') # Mentés
    print(f"Döntési határok ábrája mentve: {os.path.join(output_dir_part2, 'decision_boundaries_pca_part2.png')}")
    plt.show()


    print("\nAz 2. részfeladat alap kiértékelése, SVM hangolása és döntési határok vizualizációja befejeződött.")


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
