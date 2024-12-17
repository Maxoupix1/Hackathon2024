import pandas as pd 

# Charger les données d'entraînement 
training_data = pd.read_csv('training_data.csv') 

print("1. Données d'entraînement chargées") 

# Convertir la colonne 'time' en format datetime, si ce n'est pas déjà fait 
training_data['time'] = pd.to_datetime(training_data['time']) 

print("2. Conversion de 'time' en format datetime") 

# Rééchantillonner les données à une fréquence d'une seconde 
training_data = training_data.set_index('time').resample('1S').first().dropna().reset_index() 

# Sauvegarder le fichier avec l'extension .csv 
training_data.to_csv('training_data_second.csv', index=False) 
print("Les données rééchantillonnées ont été sauvegardées avec succès.")