import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from scipy.signal import medfilt

def segment_data(data, window_size=15*60):
    segments = []
    start_index = 0
    while start_index < len(data):
        end_index = min(start_index + window_size, len(data))
        segments.append(data.iloc[start_index:end_index])
        start_index += window_size
    return segments

print("1. Début du script")

# Charger les données d'entraînement
training_data = pd.read_csv('training_data_second.csv')
print("2. Données d'entraînement chargées")

# Sélectionner les colonnes de caractéristiques et l'étiquette
X_train = training_data[['acc_x_right', 'acc_y_right', 'acc_z_right', 'acc_x_left', 'acc_y_left', 'acc_z_left']]
y_train = training_data['activity']

# Supprimer les lignes avec des valeurs manquantes
X_train = X_train.dropna()
y_train = y_train.dropna()
print("3. Lignes avec des valeurs manquantes supprimées")

# Assurer que X et y ont des index alignés après suppression des valeurs manquantes
aligned_data = X_train.join(y_train, how='inner')

# Séparer à nouveau les caractéristiques et l'étiquette
X_train = aligned_data[['acc_x_right', 'acc_y_right', 'acc_z_right', 'acc_x_left', 'acc_y_left', 'acc_z_left']]
y_train = aligned_data['activity']
print("4. Prétraitement des données terminé")

# Définir la grille de paramètres pour la recherche
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialiser le modèle de base
clf = RandomForestClassifier(class_weight='balanced', random_state=42)

# Effectuer la recherche des meilleurs hyperparamètres
grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='f1_weighted')
grid_search.fit(X_train, y_train)
print("5. Recherche des meilleurs hyperparamètres terminée")

# Utiliser les meilleurs hyperparamètres pour entraîner le modèle final
best_clf = grid_search.best_estimator_
best_clf.fit(X_train, y_train)
print("6. Modèle entraîné avec les meilleurs hyperparamètres")

# Charger les données de prédiction
prediction_data = pd.read_csv('prediction_data.csv')
print("7. Données de prédiction chargées")

# Convertir la colonne 'time' en format datetime
prediction_data['time'] = pd.to_datetime(prediction_data['time'])
print("8. Conversion de 'time' en format datetime pour les données de prédiction")

# Rééchantillonner les données à une fréquence d'une seconde
prediction_data = prediction_data.set_index('time').resample('1S').first().dropna().reset_index()
print("9. Rééchantillonnage des données de prédiction terminé")

# Sélectionner les colonnes de caractéristiques
X_pred = prediction_data[['acc_x_right', 'acc_y_right', 'acc_z_right', 'acc_x_left', 'acc_y_left', 'acc_z_left']]

# Diviser les données de prédiction en segments de 15 minutes
segments = segment_data(X_pred, window_size=15*60)
print("10. Données de prédiction segmentées en fenêtres de 15 minutes")

# Initialiser l'encodeur
label_encoder = LabelEncoder()

# Prédictions pour chaque segment
predictions = []
for segment in segments:
    if len(segment) > 0:
        y_pred = best_clf.predict(segment)
        y_pred_encoded = label_encoder.fit_transform(y_pred)
        y_pred_smoothed_encoded = medfilt(y_pred_encoded, kernel_size=3)
        y_pred_smoothed = label_encoder.inverse_transform(y_pred_smoothed_encoded)
        predictions.extend(y_pred_smoothed)
print("11. Prédictions réalisées pour chaque segment")

# Ajouter les prédictions aux données de prédiction
prediction_data['activity'] = predictions

# Filtrer les données pour inclure celles jusqu'à 19h00:00
cutoff_time = pd.to_datetime('2024-07-25 19:00:00')
prediction_data = prediction_data[prediction_data['time'] <= cutoff_time]
print("12. Filtrage des données au-delà de 19h terminé")

# Créer la sortie au format spécifié
output = prediction_data[['time', 'activity']].copy()
output['time'] = output['time'].dt.strftime('%H:%M:%S')
output['activity'] = output['activity'].fillna('none')
print("13. Création de la sortie au format spécifié")

# Afficher les premières lignes de la sortie pour vérification
print(output.head())

# Sauvegarder la sortie dans un fichier CSV
output.to_csv('output_segmented.csv', index=False)
print("14. Les prédictions par segments ont été réalisées et le fichier output_segmented.csv a été sauvegardé avec succès.")
