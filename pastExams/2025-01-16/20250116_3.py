import numpy as np
import matplotlib.pyplot as plt

#Plot the data of the three datasets

sets = {
    "Set 1" : {
        "x1" : [0, 1, 0, -1, 0],
        "x2": [0, 0, -1, 0, 1],
        "labels": [0, 0, 0, 1, 1]
    },
    "Set 2": {
        "x1": [0, 0, -1, 1, 0],
        "x2": [0, 1, 0, 0, -1],
        "labels": [0, 0, 0, 1, 1]
    },
    "Set 3": {
        "x1": [0, 1, 0, -1, 0],
        "x2": [0, 0, 1, 0, -1],
        "labels": [1, 0, 0, 0, 0]
    }
}

def plot_data():
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, (set_name, data) in enumerate(sets.items()):
        x1, x2, labels = data["x1"], data["x2"], data["labels"]
        axs[i].scatter(
            [x for x, l in zip(x1, labels) if l == 0],
            [x for x, l in zip(x2, labels) if l == 0],
            label="Class 0",
            color="blue",
            marker="o"
        )
        axs[i].scatter(
            [x for x, l in zip(x1, labels) if l == 1],
            [x for x, l in zip(x2, labels) if l == 1],
            label="Class 1",
            color="red",
            marker="x"
        )
        axs[i].set_title(set_name)
        axs[i].set_xlabel("x1")
        axs[i].set_ylabel("x2")
        axs[i].legend()
    plt.tight_layout()
    #plt.show()

plot_data()

# ! Noting that the set 1 and 2 are linearly separable while set 3 is not.

# fint beta to use 3 as classifier for 1 and 2, threshold at 0.5

def sigmoid(z):
    return 1 / (1 + np.exp(-z)) # Defined sigmoid function

def predict(x1, x2, beta):
    return 1 if sigmoid(beta[0] + beta[1] * x1 + beta[2] * x2) >= 0.5 else -1

# Find optimal value for beta
def perception_learning_alghorimt(x1, x2, labels, eta, n_iter):
    labels = (np.array(labels) * 2) - 1  # Convert labels from {0,1} to {-1,1}
    beta = np.zeros(3)  # Initialize weights
    X = np.array([np.ones(len(x1)), x1, x2]).T # Add ones column to data matrix
    # ! BIAS TRICK \beta0 
    # TODO: IMPORTANT
    # ? BIAS TRICK: Aggiungendo una colonna di 1 alla 
    # ? matrice dei dati, possiamo trattare il termine di bias 
    # ? (intercetta) come un peso aggiuntivo. In questo modo, 
    # ? possiamo semplificare l'aggiornamento dei pesi e il calcolo 
    # ? delle predizioni, evitando di dover gestire separatamente il 
    # ? bias.
    for _ in range(n_iter):
        for i in range(len(X)):
            y_pred = predict(x1[i], x2[i], beta) # Predict
            beta += eta * (labels[i] - y_pred) * np.array([1, x1[i], x2[i]])
            # ! # beta_nuovo = beta_vecchio + learning_rate * (errore) * input
    return beta

beta1 = perception_learning_alghorimt(sets["Set 1"]["x1"], sets["Set 1"]["x2"], sets["Set 1"]["labels"], eta=0.5, n_iter=10)
beta2 = perception_learning_alghorimt(sets["Set 2"]["x1"], sets["Set 2"]["x2"], sets["Set 2"]["labels"], eta=0.5, n_iter=10)
print("Beta for Set 1:", beta1)
print("Beta for Set 2:", beta2)

# Plot (optional)
def plot_decision_boundary(beta, set_name):
    x1 = np.linspace(-1, 1, 100)
    x2 = np.linspace(-1, 1, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Y = sigmoid(beta[0] + beta[1] * X1 + beta[2] * X2)
    plt.contourf(X1, X2, Y, levels=[0, 0.5, 1], colors=['blue', 'red'], alpha=0.3)
    
    # Plot the data points
    data = sets[set_name]
    plt.scatter(data["x1"], data["x2"], c=data["labels"], cmap='bwr', edgecolors='k')
    
    plt.title(f"Decision boundary for {set_name}, x2 = {-beta[1]/beta[2]}x1 + {-beta[0]/beta[2]}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    #plt.show()

plot_decision_boundary(beta1, "Set 1")
plot_decision_boundary(beta2, "Set 2")

# ! beta is not unique, many beta can separate the data like
# ! beta1 = [0, 1, -0.5] or beta1 = [0, 2, -1] both separate set 1

# Alternative system coordinates

set = {
    "xi1": [0, 0, 0, 1, 1],
    "xi2": [0, 1, 1, 0, 0],
    "labels": [1, 0, 0, 0, 0]        
}

# Plot
plt.close('all')
fig, axs = plt.subplots(1, 1, figsize=(15, 5))
axs.scatter(
    [x for x, l in zip(set["xi1"], set["labels"]) if l == 0],
    [x for x, l in zip(set["xi2"], set["labels"]) if l == 0],
    label="Class 0",
    color="blue",
    marker="o"
)
axs.scatter(
    [x for x, l in zip(set["xi1"], set["labels"]) if l == 1],
    [x for x, l in zip(set["xi2"], set["labels"]) if l == 1],
    label="Class 1",
    color="red",
    marker="x"
)
axs.set_title("Set 3 in alternative coordinates")
axs.set_xlabel("x1")
axs.set_ylabel("x2")
axs.legend()
plt.tight_layout()
#plt.show()
# ! In questa rappresentazione, il set di dati diventa linearmente separabile.
# ! We can use the "single perception learning algorithm" (implementente via logistic regression)
# ! in order to classify the data.


# Propose a neural network to determine the parameters to be 
# used to classify set 3

# ? Rete a 2 strati:
# ? - Strato di input: 2 neuroni (xi1, xi2)
# ? - Strato nascosto: 2 neuroni, ognuno impara a calcolare un xi
# ? - Strato di output: 1 neurone con funzione di attivazione sigmoide

x1_set3 = [0, 1, 0, -1, 0]
x2_set3 = [0, 0, 1, 0, -1]
labels_e1 = [0,0,0,1,1]  # Target per il primo neurone nascosto
labels_e2 = [0,1,1,0,0]  # Target per il secondo neurone nascosto

beta_e1 = perception_learning_alghorimt(x1_set3, x2_set3, labels_e1, 0.5 , 1000)
beta_e2 = perception_learning_alghorimt(x1_set3, x2_set3, labels_e2, 0.5 , 1000)
# ? Now they can "translate" from (x1, x2) to (xi1, xi2)

# Normalizzazione output
out_1 = [(predict(x1, x2, beta_e1) + 1)/2 for x1, x2 in zip(x1_set3, x2_set3)]
out_2 = [(predict(x1, x2, beta_e2) + 1)/2 for x1, x2 in zip(x1_set3, x2_set3)]  
# ! Trucco per passare da [-1, 1] a [0, 1]
print("Output neuron 1:", out_1)
print("Output neuron 2:", out_2)

# Training ultimo neurone
beta3 = perception_learning_alghorimt(out_1, out_2, set["labels"], eta=0.01, n_iter=1000)
out_3 = [predict(xi1, xi2, beta3) for xi1, xi2 in zip(out_1, out_2)]
print("Final output:", out_3)

# Simuliamo l'intera rete 
def nn(x1, x2, beta1, beta2, beta3):
    out1 = (predict(x1, x2, beta1) + 1) / 2
    out2 = (predict(x1, x2, beta2) + 1) / 2
    print("Hidden layer outputs:", out1, out2)
    final_out = (predict(out1, out2, beta3)+1) / 2
    return final_out

out_all = [nn(x1, x2, beta_e1, beta_e2, beta3) for x1, x2 in zip(x1_set3, x2_set3)] 
print("Neural network final outputs:", out_all)