import numpy as np
import time
import os # Szükséges a mappa létrehozásához
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler # PCA előtt érdemes skálázni
import matplotlib.pyplot as plt
import seaborn as sns

# --- Mappa létrehozása az ábráknak ---
output_dir = "part1_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Létrehozva a '{output_dir}' mappa az ábrák mentéséhez.")

# --- Adatgenerálás ---
n_samples = 241
n_features = 11
centers = 4 # Ez 4 osztályt fog eredményezni
cluster_std = 0.6
random_state = 3650653441

X, y = make_blobs(n_samples=n_samples,
                  n_features=n_features,
                  centers=centers,
                  cluster_std=cluster_std,
                  random_state=random_state)

print(f"Generált adathalmaz alakja: X={X.shape}, y={y.shape}")
print(f"Osztályok eloszlása: {np.bincount(y)}")

# --- Adat vizualizációja (PCA-val 2D-ben) ---
# Skálázás PCA előtt
scaler_pca = StandardScaler()
X_scaled = scaler_pca.fit_transform(X)

pca = PCA(n_components=2, random_state=random_state)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.title('Generált adathalmaz vizualizációja (PCA után)')
plt.xlabel('Első főkomponens')
plt.ylabel('Második főkomponens')
plt.legend(handles=scatter.legend_elements()[0], labels=[f'Osztály {i}' for i in range(centers)]) # Címkék javítása
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'generated_data_pca.png')) # Mentés
plt.show()

# --- Adat felosztása tanító és teszt halmazra (eredeti adatokon) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)

# Skálázás a modellekhez (opcionális, de SVM-hez és LogReg-hez ajánlott)
scaler_model = StandardScaler()
X_train_scaled = scaler_model.fit_transform(X_train)
X_test_scaled = scaler_model.transform(X_test)


print(f"Tanító halmaz mérete: {X_train_scaled.shape[0]}")
print(f"Teszt halmaz mérete: {X_test_scaled.shape[0]}")

# --- Modellek definiálása ---
models = {
    "Naive Bayes": GaussianNB(),
    # SVC-nél a probability=True kell a későbbi predikciókhoz, ha szükséges
    "Support Vector Machine": SVC(random_state=random_state, probability=True),
    "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000) # max_iter növelése a konvergencia érdekében
}

# --- Modellek tanítása és kiértékelése ---
results = {}
training_times = {}
for name, model in models.items():
    print(f"\n--- {name} ---")
    start_time = time.time()
    # Tanítás a skálázott eredeti adatokon
    model.fit(X_train_scaled, y_train)
    end_time = time.time()
    training_times[name] = end_time - start_time

    # Jóslás (tanító és teszt adatokon)
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Kiértékelés
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    report_test = classification_report(y_test, y_pred_test)
    cm_test = confusion_matrix(y_test, y_pred_test)

    results[name] = {
        "model": model, # Mentsük el a tanított modellt is
        "accuracy_train": accuracy_train,
        "accuracy_test": accuracy_test,
        "report_test": report_test,
        "confusion_matrix_test": cm_test
    }

    print(f"Tanítási idő: {training_times[name]:.4f} másodperc")
    print(f"Pontosság (Tanító halmaz): {accuracy_train:.4f}")
    print(f"Pontosság (Teszt halmaz): {accuracy_test:.4f}")
    print("Osztályozási riport (Teszt halmaz):")
    print(report_test)
    print("Konfúziós mátrix (Teszt halmaz):")
    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=range(centers), yticklabels=range(centers))
    plt.xlabel('Jósolt osztály')
    plt.ylabel('Valós osztály')
    plt.title(f'{name} - Konfúziós Mátrix (Teszt)')
    # Fájlnév létrehozása (szóközök helyett aláhúzás)
    safe_name = name.replace(" ", "_").lower()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{safe_name}.png')) # Mentés
    plt.show()


# --- Eredmények összefoglalása ---
print("\n--- Összehasonlítás ---")
print(f"{'Modell':<25} {'Tanítási idő (s)':<20} {'Tanuló pontosság':<20} {'Teszt pontosság':<20}")
print("-" * 85)
for name, metrics in results.items():
    print(f"{name:<25} {training_times[name]:<20.4f} {metrics['accuracy_train']:<20.4f} {metrics['accuracy_test']:<20.4f}")


# --- Hiperparaméter-hangolás (Példa: SVM 'C' paramétere) ---
print("\n--- Hiperparaméter-hangolás (SVM 'C') ---")
param_grid_svm = {'C': [0.01, 0.1, 1, 10, 100]}
svm_cv = SVC(random_state=random_state, probability=True) # Új modell a GridSearchCV-hez

# GridSearchCV a legjobb C érték megtalálására a tanító adatokon (cross-validationnel)
grid_search = GridSearchCV(svm_cv, param_grid_svm, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train) # Tanítás a skálázott tanító adatokon

print(f"Legjobb 'C' paraméter: {grid_search.best_params_['C']}")
print(f"Legjobb cross-validation pontosság: {grid_search.best_score_:.4f}")

# Eredmények ábrázolása C függvényében
cv_results = grid_search.cv_results_
plt.figure(figsize=(8, 5))
plt.plot(param_grid_svm['C'], cv_results['mean_test_score'], marker='o')
plt.xscale('log')
plt.xlabel("'C' paraméter értéke")
plt.ylabel("Átlagos Cross-Validation Pontosság")
plt.title("SVM Teljesítmény a 'C' paraméter függvényében")
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'svm_c_parameter_tuning.png')) # Mentés
plt.show()

# Értékeljük ki a legjobb SVM modellt a teszt adatokon
best_svm = grid_search.best_estimator_
y_pred_svm_best = best_svm.predict(X_test_scaled)
accuracy_svm_best_test = accuracy_score(y_test, y_pred_svm_best)
print(f"\nLegjobb SVM modell pontossága a teszt halmazon: {accuracy_svm_best_test:.4f}")
print("Osztályozási riport (Legjobb SVM, Teszt halmaz):")
print(classification_report(y_test, y_pred_svm_best))


# --- Döntési határok vizualizációja (2D PCA adatokon) ---
print("\n--- Döntési határok vizualizációja (2D PCA adatokon) ---")

# Adatok felosztása és skálázása a 2D PCA adatokhoz
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=random_state, stratify=y)
# Nincs szükség külön skálázásra, mert a PCA kimenete már központosított

# Modellek újratanítása a 2D PCA adatokon
models_pca = {
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine (Best C)": best_svm, # Használjuk a hangolt SVM-et
    "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000)
}

plt.figure(figsize=(18, 5))
plot_idx = 1
all_handles = []
all_labels = []
for name, model_pca in models_pca.items():
    # Tanítás a 2D adatokon
    model_pca.fit(X_train_pca, y_train_pca)

    # Plotting decision regions
    ax = plt.subplot(1, len(models_pca), plot_idx)

    # Create meshgrid
    h = .02  # step size in the mesh
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict on meshgrid
    Z = model_pca.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot contour and training points
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    scatter_train = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_pca, cmap=plt.cm.coolwarm, s=20, edgecolors='k', label='Tanító pontok')
    scatter_test = ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test_pca, cmap=plt.cm.coolwarm, s=50, edgecolors='grey', marker='^', label='Teszt pontok')

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)

    # Collect handles and labels for the legend only once
    if plot_idx == 1:
        handles_train, labels_train = scatter_train.legend_elements()
        handles_test, labels_test = scatter_test.legend_elements(prop="colors") # Use prop="colors" for test points if needed
        all_handles.extend(handles_train)
        all_labels.extend([f'Osztály {i}' for i in range(centers)])
        # Only add one handle for all test points
        if handles_test:
             all_handles.append(scatter_test) # Use the scatter object itself for the test legend handle
             all_labels.append('Teszt pontok')


    plot_idx += 1

# Add legend to the figure, outside the last subplot
plt.figlegend(handles=all_handles, labels=all_labels, loc='center right', bbox_to_anchor=(1.05, 0.5)) # Adjust position as needed

plt.suptitle("Döntési határok a 2D PCA adatokon")
plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make room for legend
plt.savefig(os.path.join(output_dir, 'decision_boundaries_pca.png'), bbox_inches='tight') # Mentés, bbox_inches='tight' a legenda miatt
plt.show()

print("\nAz 1. részfeladat elemzése befejeződött.")