import numpy as np
import pandas as pd
from predict import loadParameters

# Supposons que y_true sont les vraies valeurs des prix et y_pred sont les valeurs prédites par votre modèle
data =  pd.read_csv('data.csv')
y_true = data['price']
t0, t1 = loadParameters()
y_pred = t0 + (t1 * data['km'])

# Calcul de la moyenne de y_true
y_mean = np.mean(y_true)

# Calcul de la somme des carrés totaux (SST)
sst = np.sum((y_true - y_mean)**2)

# Calcul de la somme des carrés des résidus (SSE)
sse = np.sum((y_true - y_pred)**2)

# Calcul du coefficient de détermination R²
r_squared = 1 - (sse / sst)

print("Coefficient de détermination (R²) :", r_squared)

# Calcul de l'erreur quadratique moyenne (RMSE)
rmse = np.sqrt(np.mean((y_pred - y_true)**2))

# Calcul de l'erreur absolue moyenne (MAE)
mae = np.mean(np.abs(y_pred - y_true))

print("Erreur quadratique moyenne (RMSE) :", rmse)
print("Erreur absolue moyenne (MAE) :", mae)

# Calcul de l'erreur quadratique moyenne (MSE)
mse = np.mean((y_pred - y_true)**2)

print("Mean Squared Error (MSE) :", mse)