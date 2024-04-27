from predict import estimatePrice
import pandas as pd
import numpy as np


def precision() -> None:
    """
    Calculate and print the precision of the model

    Args:
        None

    Returns:
        None
    """

    df = pd.read_csv('data.csv')
    km = df['km'].values
    price = df['price'].values

    for i in range(len(km)):
        price_estimated = estimatePrice(km[i])
        price_real = price[i]
        precision = 100 - (abs(price_estimated - price_real) / price_real * 100)
        if i == 0:
            precision_min = precision
            precision_max = precision
            precision_moy = precision
        else:
            if precision < precision_min:
                precision_min = precision
            if precision > precision_max:
                precision_max = precision
            precision_moy += precision

    precision_moy /= len(km)

    print(f"La précision minimum du modèle est de: {precision_min:.2f}%")
    print(f"La précision moyenne du modèle est de: {precision_moy:.2f}%")
    print(f"La précision maximal du modèle est de: {precision_max:.2f}%")


if __name__ == "__main__":
    precision()