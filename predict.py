import pandas as pd
import numpy as np


def loadParameters() -> tuple:
    """ 
    Load the parameters from the file theta.csv

    Returns:
        float: The value of theta0
        float: The value of theta1
    """
    try:
        df = pd.read_csv('theta.csv')
        return df['theta0'][0], df['theta1'][0]
    except Exception as e:
        print(f"Erreur dans le chargement de θ₀ et θ₁, initialisation à 0.")
        return 0, 0


def priceDenormalization(price: float, price_values: np.ndarray) -> float:
    """
    Denormalize the price of a car
    
    Args:
        price (float): The price of the car
        price_values (np.ndarray): The prices of the cars
        
    Returns:
        float: The denormalized price of the car
    """
    max = np.max(price_values)
    min = np.min(price_values)
    return price * (max - min) + min


def kmNormalization(km: float, km_values: np.ndarray) -> float:
    """
    Normalize the km of a car
    
    Args:
        km (float): The km of the car
        km_values (np.ndarray): The km of the cars
        
    Returns:
        float: The normalized km of the car
    """
    max = np.max(km_values)
    min = np.min(km_values)
    return (km - min) / (max - min)


def estimatePrice(km: float) -> float:
    """
    Estimate the price of a car given its km
    
    Args:
        km (float): The km of the car
        
    Returns:
        float: The estimated price of the car
    """
    dataFrame = pd.read_csv('data.csv')
    price_values = dataFrame['price'].values
    km_values = dataFrame['km'].values
    t0, t1 = loadParameters()
    if t0 == 0 and t1 == 0:
        return 0
    else: 
        return priceDenormalization(t0 + (t1 * kmNormalization(km, km_values)), price_values)


if __name__ == "__main__":
    km = float(input("Entrez le kilométrage de la voiture : "))
    print(f"Le prix estimé de la voiture pour un kilométrage de {km} miles est : {estimatePrice(km):.2f} euros.")