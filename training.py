import numpy as np

# Fonction pour lire les données à partir du fichier CSV
def readData(filename):
    with open(filename, 'r') as file:
        next(file)  # Ignorer l'en-tête
        data = [list(map(int, line.strip().split(','))) for line in file]
    return data

def saveParameters(theta0, theta1):
    with open("parameters.txt", 'w') as file:
        file.write("{},{}".format(0, 0))

def estimatePrice(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

# Fonction pour entraîner le modèle et sauvegarder les paramètres
def trainModel(data, learning_rate):
    
    saveParameters(0, 0)
    theta0, theta1 = 0, 0
    
    m = len(data)

    for _ in range(1000):
        tmp_theta0, tmp_theta1 = 0, 0
        for mileage, price in data:
            tmp_theta0 += (estimatePrice(mileage, theta0, theta1) - price)
            tmp_theta1 += (estimatePrice(mileage, theta0, theta1) - price) * mileage
        
        theta0 -= learning_rate * 1/m * tmp_theta0
        theta1 -= learning_rate * 1/m * tmp_theta1

        print("theta0: {}, theta1: {}".format(theta0, theta1))


    with open("parameters.txt", 'w') as file:
        file.write("{},{}".format(theta0, theta1))


if __name__ == "__main__":
    data = readData("data.csv")
    learning_rate = 0.1 # Taux d'apprentissage
    
    trainModel(data, learning_rate)
    