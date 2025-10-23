import pennylane as qml
import numpy as np
from sklearn.preprocessing import normalize

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # Encode input into qubits
    for i in range(n_qubits):
        qml.RY(inputs[i % len(inputs)], wires=i)
    # Variational layer
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
        qml.CNOT(wires=[i, (i+1)%n_qubits])
    return qml.expval(qml.PauliZ(0))

def quantum_predict(feature_vector, weights):
    vec = np.array(feature_vector)
    vec = normalize(vec.reshape(1, -1))[0] * np.pi
    return (quantum_circuit(vec, weights) + 1)/2

def quantum_ner_text(text, vectorizer):
    """
    Returns tokenized entities and quantum scores
    """
    weights = np.random.uniform(0, np.pi, n_qubits)
    entities = text.split()
    scores = []
    for word in entities:
        vec = vectorizer.transform([word]).toarray()[0]
        score = quantum_predict(vec, weights)
        scores.append(round(float(score), 2))
    return entities, scores
