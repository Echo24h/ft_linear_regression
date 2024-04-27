import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def saveParameters(t0: float, t1: float) -> None:
    """
    Sauvegarde les paramètres dans un fichier theta.csv

    Args:
        t0 (float): La valeur de θ₀
        t1 (float): La valeur de θ₁
    
    Returns:
        None
    """
    pd.DataFrame({'theta0': [t0], 'theta1': [t1]}).to_csv('theta.csv', index=False)


def dataNormalization(data: np.ndarray) -> np.ndarray:
    """
    Normalise les données
    
    Args:
        data (array): Les données à normaliser
        
    Returns:
        array: Les données normalisées
    """
    max = np.max(data)
    min = np.min(data)
    return (data - min) / (max - min)


def dataDeNormalization(data: np.ndarray, original_data: np.ndarray) -> np.ndarray:
    """
    Dénormalise les données
    
    Args:
        data (array): Les données à dénormaliser
        original_data (array): Les données originales
        
    Returns:
        array: Les données dénormalisées
    """
    max = np.max(original_data)
    min = np.min(original_data)
    return data * (max - min) + min


def trainModel(km: np.ndarray, price: np.ndarray, learning_rate: float, epochs: int) -> tuple:
    """
    Entraîne le modèle avec un taux d'apprentissage donné

    Args:
        km (array): Les kilomètres des voitures
        price (array): Les prix des voitures
        learning_rate (float): Le taux d'apprentissage
        epochs (int): Le nombre d'itérations
    
    Returns:
        float: La valeur de θ₀
        float: La valeur de θ₁
    """

    n = len(km)

    t0, t1 = 0, 0
    
    for _ in range(epochs):
        prediction = t0 + (t1 * km)
        t0 -= learning_rate / n * sum(prediction - price)
        t1 -= learning_rate / n * sum((prediction - price) * km) 

    return t0, t1


def main() -> None:

    dataFrame = pd.read_csv('data.csv')
    km = dataFrame['km'].values
    price = dataFrame['price'].values

    learning_rates = 0.1
    epochs = 1000

    km_normalized = dataNormalization(km)
    price_normalized = dataNormalization(price)

    # Train the model with the normalized data
    t0, t1 = trainModel(km_normalized, price_normalized, learning_rates, epochs)

    print(f"Learning rate: {learning_rates}")
    print(f"θ₀: {t0}")
    print(f"θ₁: {t1}")

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the results normalized
    axs[0].plot(km_normalized, t0 + (t1 * km_normalized), color='red', linewidth=1)
    axs[0].scatter(km_normalized, price_normalized)
    axs[0].set_title('Price prediction')
    axs[0].set_xlabel('km (normalized)')
    axs[0].set_ylabel('Price (normalized)')

    # Plot the results de-normalized
    axs[1].plot(km, dataDeNormalization(t0 + (t1 * km_normalized), price), color='red', linewidth=1)
    axs[1].scatter(km, price)
    axs[1].set_title('Price prediction')
    axs[1].set_xlabel('km')
    axs[1].set_ylabel('Price')

    plt.tight_layout()
    plt.show()

    saveParameters(t0, t1)

if __name__ == "__main__":
    main()