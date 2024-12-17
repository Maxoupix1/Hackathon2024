import pandas as pd

data = pd.read_csv('output.csv')

print(data['activity'].value_counts())
print(len(data))

data = pd.read_csv('prediction_output.csv')

print(data['activity'].value_counts())
print(len(data))

data = pd.read_csv('output_segmented.csv')

print(data['activity'].value_counts())
print(len(data))