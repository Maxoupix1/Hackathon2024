import pandas as pd
from claspy.segmentation import BinaryClaSPSegmentation
import numpy as np

# Charger les données
training_data = pd.read_csv('training_data_second.csv')
X = training_data[['acc_x_right', 'acc_y_right', 'acc_z_right', 'acc_x_left', 'acc_y_left', 'acc_z_left']].values

# Normaliser les données
X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Appliquer la segmentation avec BinaryClaSPSegmentation
segmenter = BinaryClaSPSegmentation()
segments = segmenter.fit_predict(X_normalized)

# Ajouter les segments aux données
training_data['segment'] = segments

# Afficher les segments
print(training_data.head())

# Sauvegarder les données segmentées
training_data.to_csv('training_data_segmented.csv', index=False)
print('Les données segmentées ont été sauvegardées dans le fichier training_data_segmented.csv.')

import matplotlib.pyplot as plt

# Visualiser les segments
plt.figure(figsize=(12, 6))
for i in range(6):  # Visualiser chaque caractéristique
    plt.subplot(6, 1, i+1)
    plt.plot(training_data.index, training_data.iloc[:, i], label=training_data.columns[i])
    plt.scatter(training_data.index, training_data['segment'], c=training_data['segment'], cmap='viridis', label='Segments', marker='|')
    plt.legend(loc='best')
plt.tight_layout()
plt.show()
