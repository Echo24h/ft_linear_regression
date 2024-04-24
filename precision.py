from predict import loadParameters
import pandas as pd
import numpy as np



def precision() -> None:
    """
    Calculate the precision of the model

    Args:
        None

    Returns:
        None
    """
    t0, t1 = loadParameters()
    dataFrame = pd.read_csv('data.csv')

    km = dataFrame['km']
    price = dataFrame['price']

    precision_min = 100 - max(abs(price - (t0 + (t1 * km))) / price * 100)
    precision_max = 100 - min(abs(price - (t0 + (t1 * km))) / price * 100)
    precision_moy = 100 - np.mean(abs(price - (t0 + (t1 * km))) / price * 100)

    print(f"θ₀: {t0}")
    print(f"θ₁: {t1}")

    print(f"La précision minimum du modèle est de: {precision_min:.2f}%")
    print(f"La précision moyenne du modèle est de: {precision_moy:.2f}%")
    print(f"La précision maximal du modèle est de: {precision_max:.2f}%")


if __name__ == "__main__":
    precision()