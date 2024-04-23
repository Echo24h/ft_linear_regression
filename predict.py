
def loadParameters(filename):
    with open(filename, 'r') as file:
        theta0, theta1 = map(float, file.readline().strip().split(','))
    return theta0, theta1

# Fonction pour prédire le prix en utilisant les paramètres theta0 et theta1
def estimatePrice(mileage):
    theta0, theta1 = loadParameters("parameters.txt")
    return theta0 + (theta1 * mileage)


if __name__ == "__main__":
    mileage = float(input("Entrez le kilométrage de la voiture : "))
    estimated_price = estimatePrice(mileage)
    print("Le prix estimé de la voiture pour un kilométrage de {} km est : {:.2f} euros.".format(mileage, estimated_price))