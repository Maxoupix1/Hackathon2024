import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

# 1. Charger les fichiers CSV et Excel
left_accs = pd.read_csv('left_accs.csv', header=None, names=['time', 'x_left', 'y_left', 'z_left'])
right_accs = pd.read_csv('right_accs.csv', header=None, names=['time', 'x_right', 'y_right', 'z_right'])
activities = pd.read_excel('activite_update-2.xlsx')

# Définir les bornes de la série temporelle
start_time = datetime(left_accs['time'].min())
end_time = datetime(left_accs['time'].max())

# 2. Transformer les timestamps Unix en jours, secondes et millisecondes
def convert_time(timestamp):
    dt = datetime.utcfromtimestamp(timestamp)
    return dt.strftime('%Y-%m-%d'), dt.strftime('%H:%M:%S'), int(dt.microsecond / 1000)

print(1)
left_accs[['date', 'time_of_day', 'millisecond']] = left_accs['time'].apply(
    lambda t: pd.Series(convert_time(t))
)
print(2)
# 3. Interpoler les données de la main droite
# Fixer les temps de la main gauche comme référence
time_left = left_accs['time']
right_accs_interpolated = pd.DataFrame()
right_accs_interpolated['time'] = time_left
right_accs_interpolated['x_right'] = np.interp(time_left, right_accs['time'], right_accs['x_right'])
right_accs_interpolated['y_right'] = np.interp(time_left, right_accs['time'], right_accs['y_right'])
right_accs_interpolated['z_right'] = np.interp(time_left, right_accs['time'], right_accs['z_right'])
print(3)
# 4. Associer les activités
# Ajouter une colonne "Activité" par correspondance entre t1 <= temps <= t2
left_accs['Activite'] = 'none'  # Initialiser une colonne vide pour les activités
for _, row in tqdm(activities.iterrows(), total=activities.shape[0], desc="Assigning activities"):
    t1, t2, activity = row['t1'], row['t2'], row['Activité']
    mask = (left_accs['time'] >= t1) & (left_accs['time'] <= t2)
    left_accs.loc[mask, 'Activite'] = activity

# 5. Fusionner les données de la main gauche, main droite, et activités
final_df = pd.concat([
    left_accs[['date', 'time_of_day', 'millisecond', 'x_left', 'y_left', 'z_left', 'Activite']],
    right_accs_interpolated[['x_right', 'y_right', 'z_right']]
], axis=1)

# 6. Créer les DataFrames d'entraînement et de prédiction
cutoff_time = datetime(2024, 7, 25, 7, 0, 0).timestamp()

# DataFrame d'entraînement
train_df = final_df[final_df['time'] <= cutoff_time][['x_left', 'y_left', 'z_left', 'x_right', 'y_right', 'z_right', 'Activite']]

# DataFrame de prédiction
predict_df = final_df[(final_df['time'] > cutoff_time) & (final_df['time'] <= datetime(2024, 7, 25, 19, 0, 0).timestamp())][['x_left', 'y_left', 'z_left', 'x_right', 'y_right', 'z_right']]

# 7. Sauvegarder ou afficher
#train_df.to_csv('train_data.csv', index=False)
#predict_df.to_csv('predict_data.csv', index=False)

print("DataFrame d'entraînement :")
print(train_df.head())
print(train_df.tail())
print("\nDataFrame de prédiction :")
print(predict_df.head())
print(predict_df.tail())

