import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf

tf.autograph.set_verbosity(0)
import warnings

warnings.filterwarnings('ignore')
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
import pennylane as qml
import Tool

np.random.seed(52)


def QCNNcircuit(inputs, QC1, QC2, QC3, QC4, QC5, QC6,
                QP1, QP2, QP3, QP4, QP5, QP6,
                QF):
    # print(inputs.shape)
    start_qubits = 7
    qml.AmplitudeEmbedding(inputs, wires=range(start_qubits), normalize=True)  # encoder
    QC = [QC1, QC2, QC3, QC4, QC5, QC6]
    QP = [QP1, QP2, QP3, QP4, QP5, QP6]
    qcstep = [4, 1, 2, 1, 1, 1]
    # region
    for j in range(6):
        sq = start_qubits - j
        qc = QC[j]
        qp = QP[j]
        step = qcstep[j]

        index = 0
        for i in range(sq):
            qml.RY(qc[index], wires=i)
            index = index + 1
        for i in range(sq):
            if (i == 0):
                qml.RY(qc[index], wires=i)
                index = index + 1
                qml.CNOT(wires=[sq - 1, 0])
                qml.RY(qc[index], wires=i)
                index = index + 1
            else:
                qml.RY(qc[index], wires=sq - i)
                index = index + 1
                qml.CNOT(wires=[sq - i - 1, sq - i])
                qml.RY(qc[index], wires=sq - i)
                index = index + 1
        for i in range(sq):
            qml.RY(qc[index], wires=i)
            index = index + 1
        # print(index)
        control = 0
        controled = step + control
        for i in range(sq):
            qml.RY(qc[index], wires=controled)
            index = index + 1
            qml.CNOT(wires=[control, controled])
            qml.RY(qc[index], wires=controled)
            index = index + 1
            control = controled
            controled = (controled + step) % (sq)
        index = 0

        for i in range(sq - 1):
            qml.RY(qp[index], wires=i + 1)
            index = index + 1
            qml.CNOT(wires=[i, i + 1])
            qml.RY(qp[index], wires=i + 1)
            index = index + 1
    qml.RY(QF, wires=0)
    return qml.probs(wires=[0])


class MyQCNNLayer(tf.keras.layers.Layer):

    def __init__(self, start_qubits, final_qubits):
        super(MyQCNNLayer, self).__init__()
        self.kernel = None
        self.start_qubits = start_qubits
        self.final_qubits = final_qubits
        self.QC1 = tf.Variable(tf.zeros(6 * start_qubits, dtype=tf.float64))  # 42
        self.QP1 = tf.Variable(tf.zeros(2 * (start_qubits - 1), dtype=tf.float64))  # 12
        self.QC2 = tf.Variable(tf.zeros(6 * (start_qubits - 1), dtype=tf.float64))  # 36
        self.QP2 = tf.Variable(tf.zeros(2 * (start_qubits - 2), dtype=tf.float64))  # 10
        self.QC3 = tf.Variable(tf.zeros(6 * (start_qubits - 2), dtype=tf.float64))  # 30
        self.QP3 = tf.Variable(tf.zeros(2 * (start_qubits - 3), dtype=tf.float64))  # 8
        self.QC4 = tf.Variable(tf.zeros(6 * (start_qubits - 3), dtype=tf.float64))  # 24
        self.QP4 = tf.Variable(tf.zeros(2 * (start_qubits - 4), dtype=tf.float64))  # 6
        self.QC5 = tf.Variable(tf.zeros(6 * (start_qubits - 4), dtype=tf.float64))  # 18
        self.QP5 = tf.Variable(tf.zeros(2 * (start_qubits - 5), dtype=tf.float64))  # 4
        self.QC6 = tf.Variable(tf.zeros(6 * (start_qubits - 5), dtype=tf.float64))  # 12
        self.QP6 = tf.Variable(tf.zeros(2 * (start_qubits - 6), dtype=tf.float64))  # 2

        self.QF = tf.Variable(0, dtype=tf.float64)  # 1
        self.dev = qml.device("default.qubit", wires=self.start_qubits)
        self.qnode = qml.QNode(QCNNcircuit, self.dev, interface='tf')

    def call(self, inputs):
        out = []
        inputs = tf.squeeze(inputs)
        # print(inputs.shape)
        for i in range(len(inputs)):
            try:
                out_temp = self.qnode(inputs[i],
                                      self.QC1, self.QC2, self.QC3, self.QC4, self.QC5, self.QC6,
                                      self.QP1, self.QP2, self.QP3, self.QP4, self.QP5, self.QP6,
                                      self.QF)

                # print(out_temp)
            except:
                print(inputs[i])

            out.append(out_temp)
        out = tf.stack(out, axis=0)
        # print(out.shape)
        return out


def grad(model, inputs, y_true):
    with tf.GradientTape() as tape:
        y_pred = model(inputs, training=True)
        # print(y_pred)
        loss_value = loss_object(y_true=y_true, y_pred=y_pred)

    y_pred = tf.argmax(y_pred, axis=1)
    right = 0
    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i]:
            right = right + 1
    return loss_value, tape.gradient(loss_value, model.trainable_variables), float(right) / len(y_true)


def evaluate_acc_loss(y_true, y_pred):
    loss = loss_object(y_pred=y_pred, y_true=y_true).numpy()
    y_pred = tf.argmax(y_pred, axis=1)
    right = 0
    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i]:
            right = right + 1
    return float(right) / len(y_true), loss


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.035)

if __name__ == "__main__":
    try:
        with tf.device('/device:GPU:1'):

            train_data, train_label, test_data, test_label = Tool.readdata_2classes_15()
            MyModel = MyQCNNLayer(start_qubits=7, final_qubits=1)
            epoch = 30
            batch_size = 100
            total_step = int(len(train_data) / batch_size)

            test_accuracy_results = []
            test_loss_results = []
            for e in range(epoch):
                for s in range(total_step):
                    x, y = train_data[s * batch_size:(s + 1) * batch_size], train_label[
                                                                            s * batch_size:(s + 1) * batch_size]
                    loss_value, grads, step_acc = grad(MyModel, x, y)
                    optimizer.apply_gradients(zip(grads, MyModel.trainable_variables))

                    print("Epoch {:03d},Step {}:Loss: {:.3f}, Accuracy: {:.3%}".format(e + 1, s + 1, loss_value.numpy(),
                                                                                       step_acc))


                test_label_pred = MyModel(test_data)
                test_acc, test_loss = evaluate_acc_loss(y_pred=test_label_pred, y_true=test_label)
                test_accuracy_results.append(test_acc)
                test_loss_results.append(test_loss)

                print("Epoch {:03d},:Loss: {:.3f}, Accuracy: {:.3%}".format(e + 1, test_loss, test_acc))
                with open("DQCNN_2Code_Acc.txt", 'w') as file:

                    file.write(str(test_accuracy_results) + "\n")
                with open("DQCNN_2Code_Loss.txt", 'w') as file:
                    file.write(str(test_loss_results) + "\n")
    except RuntimeError as e:
        print(e)
