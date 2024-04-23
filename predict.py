import pandas as pd


def loadParameters() -> tuple:
    """ 
    Load the parameters from the file theta.csv
    
    Returns:
        float: The value of theta0
        float: The value of theta1
    """
    df = pd.read_csv('theta.csv')
    if df is None or df.empty:
        print("Error: No parameters found")
        return 0, 0
    return df['theta0'][0], df['theta1'][0]


def estimatePrice(km: float) -> float:
    """
    Estimate the price of a car given its km
    
    Args:
        km (float): The km of the car
        
    Returns:
        float: The estimated price of the car
    """
    t0, t1 = loadParameters()
    return t0 + (t1 * km)


def main():
    km = float(input("Entrez les miles de la voiture : "))
    print(f"Le prix estimé de la voiture pour un kilométrage de {km} miles est : {estimatePrice(km):.2f} euros.")


if __name__ == "__main__":
    main()