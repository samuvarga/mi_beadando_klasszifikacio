import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

# --- Adat felosztása tanító és teszt halmazra ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

print(f"Tanító halmaz mérete: {X_train.shape[0]}")
print(f"Teszt halmaz mérete: {X_test.shape[0]}")

# --- Modellek definiálása ---
models = {
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(random_state=random_state),
    "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000) # max_iter növelése a konvergencia érdekében
}

# --- Modellek tanítása és kiértékelése ---
results = {}
for name, model in models.items():
    print(f"\n--- {name} ---")
    # Tanítás
    model.fit(X_train, y_train)

    # Jóslás
    y_pred = model.predict(X_test)

    # Kiértékelés
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {"accuracy": accuracy, "report": report, "confusion_matrix": cm}

    print(f"Pontosság (Accuracy): {accuracy:.4f}")
    print("Osztályozási riport:")
    print(report)
    print("Konfúziós mátrix:")
    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(centers), yticklabels=range(centers))
    plt.xlabel('Jósolt osztály')
    plt.ylabel('Valós osztály')
    plt.title(f'{name} - Konfúziós Mátrix')
    plt.show()


# --- Eredmények összefoglalása ---
print("\n--- Összehasonlítás ---")
for name, metrics in results.items():
    print(f"{name}: Pontosság = {metrics['accuracy']:.4f}")

# Itt további összehasonlításokat is végezhetsz, pl. F1-score alapján.