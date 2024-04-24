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


def dataStandardization(data: np.ndarray) -> np.ndarray:
    """
    Standardise les données
    
    Args:
        data (array): Les données à standardiser
    
    Returns:
        array: Les données standardisées
    """
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


def dataDenormalization(data: np.ndarray, original_data: np.ndarray) -> np.ndarray:
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


def dataDestandardization(data: np.ndarray, original_data: np.ndarray) -> np.ndarray:
    """
    Déstandardise les données
    
    Args:
        data (array): Les données à déstandardiser
        original_data (array): Les données originales
        
    Returns:
        array: Les données déstandardisées
    """
    mean = np.mean(original_data)
    std = np.std(original_data)
    return data * std + mean


def trainModelWithLearningRate(km: np.ndarray, price: np.ndarray, learning_rate: float, epochs: int) -> tuple:
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

    km_norm = (km / np.std(km))

    t0, t1 = 0, 0
    

    for _ in range(epochs):
        prediction = t0 + (t1 * km_norm)
        t0 -= learning_rate / n * sum(prediction - price)
        t1 -= learning_rate / n * sum((prediction - price) * km_norm) 
    
    t1 = t1 / np.std(km)

    return t0, t1


def trainModels(km: np.ndarray, price: np.ndarray) -> None:
    """
    Entraîne le modèle avec différents taux d'apprentissage et affiche les résultats
    
    Args:
        km (array): Les kilomètres des voitures
        price (array): Les prix des voitures
        
    Returns:
        None
    """

    learning_rates = {0.1, 0.01, 0.05} # Taux d'apprentissage
    epochs = 1000 # Nombre d'itérations

    plt.scatter(km, price)
    plt.title('Price = θ₀ + θ₁ * km')
    legend = ['Data']

    i = 0
    for lr in learning_rates:
            
        t0, t1 = trainModelWithLearningRate(km, price, lr, epochs)

        if i == 0: plt.plot(km, t0 + (t1 * km), color='red', linewidth=1)
        elif i == 1: plt.plot(km, t0 + (t1 * km), color='green', linewidth=1)
        elif i == 2: plt.plot(km, t0 + (t1 * km), color='orange', linewidth=1)

        legend += [f"lr = {lr}, θ₀ = {t0:.1f}, θ₁ = {t1:.5f}"]

        i += 1
        
        print(f"Learning rate: {lr}")
        print(f"θ₀: {t0}")
        print(f"θ₁: {t1}")
        print("")

        saveParameters(t0, t1)

    plt.legend(legend)
    plt.xlabel('km')
    plt.ylabel('Price')
    plt.show()


def main():
    dataFrame = pd.read_csv('data.csv')
    km = dataFrame['km'].values
    price = dataFrame['price'].values
    trainModels(km, price)

if __name__ == "__main__":
    main()