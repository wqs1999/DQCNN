import numpy as np


def generate_data(qubits):
    """Generate training and testing data."""
    n_rounds = 80  # Produces n_rounds * n_qubits datapoints.
    data = []
    for n in range(n_rounds):
        for bit in range(qubits):
            l=[]
            rng = np.random.uniform(-np.pi, np.pi)
            l.append(bit)
            l.append(rng)
            l.append(1 if (-np.pi / 2) <= rng <= (np.pi / 2) else 0)
            data.append(l)

    data = np.array(data)
    np.random.shuffle(data)
    split_ind = int(len(data) * 0.8)
    train_data = data[:split_ind]
    test_data = data[split_ind:]

    train_data, train_label = np.hsplit(train_data, [2])
    test_data, test_label = np.hsplit(test_data, [2])

    train_label = np.squeeze(train_label)
    test_label = np.squeeze(test_label)
    return train_data, train_label, test_data, test_label



if __name__ == "__main__":
    train_data, train_label, test_data, test_label = generate_data(7)
    print(test_data)
    print(train_label.shape)
    print(test_label.shape)
