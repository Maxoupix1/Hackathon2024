import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score
from scipy.signal import medfilt
from sklearn.preprocessing import LabelEncoder

# Charger les données d'entraînement
training_data = pd.read_csv('training_data_second.csv')

print("3. Rééchantillonnage terminé")
print(training_data.head())

# Sélectionner les colonnes de caractéristiques et l'étiquette
X = training_data[['acc_x_right', 'acc_y_right', 'acc_z_right', 'acc_x_left', 'acc_y_left', 'acc_z_left']]
y = training_data['activity']

# Supprimer les lignes avec des valeurs manquantes
X = X.dropna()
y = y.dropna()

# Assurer que X et y ont des index alignés après suppression des valeurs manquantes
aligned_data = X.join(y, how='inner')

# Séparer à nouveau les caractéristiques et l'étiquette
X = aligned_data[['acc_x_right', 'acc_y_right', 'acc_z_right', 'acc_x_left', 'acc_y_left', 'acc_z_left']]
y = aligned_data['activity']

print("4. Prétraitement des données terminé")

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("5. Division des données terminé")

# Entraîner le modèle
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print("6. Modèle entraîné")

# Prédictions sur l'ensemble de test
y_pred_test = clf.predict(X_test)

# Calculer le F𝛽-measure avec 𝛽 = 1/3
fbeta_test = fbeta_score(y_test, y_pred_test, beta=1/3, average='weighted')

# Afficher la matrice de confusion et le rapport de classification pour l'ensemble de test
print("7. Evaluation sur l'ensemble de test")
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred_test))
print("Rapport de classification :\n", classification_report(y_test, y_pred_test))
print(f'F𝛽-measure (𝛽=1/3) : {fbeta_test:.4f}')

# Charger les données de prédiction
prediction_data = pd.read_csv('prediction_data.csv')

print("8. Données de prédiction chargées")

# Convertir la colonne 'time' en format datetime
prediction_data['time'] = pd.to_datetime(prediction_data['time'])

print("9. Conversion de 'time' en format datetime pour les données de prédiction")

# Rééchantillonner les données à une fréquence d'une seconde
prediction_data = prediction_data.set_index('time').resample('1S').first().dropna().reset_index()

print("10. Rééchantillonnage des données de prédiction terminé")

# Sélectionner les colonnes de caractéristiques
X_pred = prediction_data[['acc_x_right', 'acc_y_right', 'acc_z_right', 'acc_x_left', 'acc_y_left', 'acc_z_left']]

# Prédictions sur l'ensemble de prédiction
y_pred = clf.predict(X_pred)

print("11. Prédictions sur l'ensemble de prédiction terminées")

# Encoder les prédictions pour appliquer le filtre médian
label_encoder = LabelEncoder()
y_pred_encoded = label_encoder.fit_transform(y_pred)

# Appliquer un filtre médian pour lisser les prédictions
y_pred_smoothed_encoded = medfilt(y_pred_encoded, kernel_size=3)

# Décoder les prédictions lissées
y_pred_smoothed = label_encoder.inverse_transform(y_pred_smoothed_encoded)

# Ajouter les prédictions lissées aux données de prédiction
prediction_data['activity'] = y_pred_smoothed

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
output.to_csv('prediction_output.csv', index=False)

print('Les prédictions ont été réalisées et le fichier a été sauvegardé avec succès.')
