import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv('./dataTrain.csv')

X = df['card'].values.reshape(-1, 1)
y = df['isFraudulent']

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)


print(f'Accuracy: {model.score(X_test, y_test)}')


pickle.dump(model, open('app/models/eater.pkl', 'wb'))
