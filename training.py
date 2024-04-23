import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def saveParameters(t0, t1):
    pd.DataFrame({'theta0': [t0], 'theta1': [t1]}).to_csv('theta.csv', index=False)


def trainModelWithLearningRate(km, price, t0, t1, learning_rate, epochs):

    n = len(km)

    km_mean = np.mean(km)

    km_norm = (km / km_mean)# / np.std(km)
    

    for _ in range(epochs):
        prediction = t0 + (t1 * km)
        t0 -= learning_rate / n * sum(prediction - price)
        t1 -= learning_rate / n * sum((prediction - price) * km_norm)

    t1 = t1 / km_mean

    return t0, t1


# Fonction pour entraîner le modèle et sauvegarder les paramètres
def trainModel(km, price):

    learning_rates = {1, 0.1, 0.01} # Taux d'apprentissage
    epochs = 100 # Nombre d'itérations

    t0, t1 = 0, 0

    for lr in learning_rates:
            
            t0, t1 = trainModelWithLearningRate(km, price, t0, t1, lr, epochs)
            print(f"Learning rate: {lr}")
            print(f"θ₀: {t0}")
            print(f"θ₁: {t1}")
            print("Price = θ₀ + θ₁ * km")
            print("")

            saveParameters(t0, t1)

    # for _ in range(epochs):
    #    prediction = t0 + (t1 * km_norm)
    #    error[_] = sum((prediction - price) ** 2) / (2 * n)
    #    t0 -= learning_rate / n * sum(prediction - price)
    #    t1 -= learning_rate / n * sum((prediction - price) * km_norm)
        
    # plt.plot(list(error.keys()), list(error.values()))
    # plt.title('Error')
    # plt.show()

    #t1 = t1 / np.std(km)
    

    return t0, t1


def main():
    dataFrame = pd.read_csv('data.csv')

    

    km = dataFrame['km'].values
    price = dataFrame['price'].values

    t0, t1 = trainModel(km, price)

    print(f"θ₀: {t0}")
    print(f"θ₁: {t1}")
    print("Price = θ₀ + θ₁ * km")

    plt.scatter(km, price)

    plt.labelx = 'km'
    plt.labely = 'price'

    plt.title('Price = θ₀ + θ₁ * km')
    plt.legend(['Data', ''])

    plt.plot(km, t0 + (t1 * km), color='red')
    plt.show()



if __name__ == "__main__":
    main()