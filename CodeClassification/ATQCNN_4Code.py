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


def ATQCNNcircuit(inputs, QC1, QC2, QC3, QP1, QP2, QP3, QF):

    qml.AmplitudeEmbedding(inputs, wires=range(7), normalize=True)  # encoder 前面七个编码
    qubits_f = [0, 1, 2, 3, 4, 5, 6]
    # region 第一层卷积
    for i in range(7):
        para_index = 0
        for q in range(6):
            qml.RY(QC1[para_index], wires=i)
            para_index = para_index + 1
            qml.RY(QC1[para_index], wires=(i + 1) % 7)
            para_index = para_index + 1
            qml.CNOT(wires=[i, (i + 1) % 7])
    # endregion
    # region 第一层池化
    for i in range(3):
        para_index = 0
        qml.RY(QP1[para_index], wires=i)
        para_index = para_index + 1
        qml.RY(QP1[para_index], wires=i+3)
        qml.CNOT(wires=[i, (i + 3)])
        qml.RY(-QP1[para_index], wires=i+3)
    # endregion

    # region 第二层卷积
    for i in range(3,7):
        para_index = 0
        if i!=6:
            for q in range(6):
                qml.RY(QC2[para_index], wires=i)
                para_index = para_index + 1
                qml.RY(QC2[para_index], wires=(i + 1) % 7)
                para_index = para_index + 1
                qml.CNOT(wires=[i, (i + 1) % 7])
        if i==6:
            for q in range(6):
                qml.RY(QC2[para_index], wires=i)
                para_index = para_index + 1
                qml.RY(QC2[para_index], wires=(i + 4) % 7)
                para_index = para_index + 1
                qml.CNOT(wires=[i, (i + 1) % 7])
    # endregion
    # region 第二层池化
    for i in range(3,5):
        para_index = 0
        qml.RY(QP2[para_index], wires=i)
        para_index = para_index + 1
        qml.RY(QP2[para_index], wires=i + 2)
        qml.CNOT(wires=[i, (i + 2)])
        qml.RY(-QP2[para_index], wires=i + 2)
    # endregion

    # region 第三层卷积
    para_index = 0
    for q in range(6):
        qml.RY(QC3[para_index], wires=5)
        para_index = para_index + 1
        qml.RY(QC3[para_index], wires=6)
        para_index = para_index + 1
        qml.CNOT(wires=[i, (i + 1) % 7])
    # endregion

    # region 第三层池化
    para_index = 0
    qml.RY(QP3[para_index], wires=5)
    para_index = para_index + 1
    qml.RY(QP3[para_index], wires=6)
    qml.CNOT(wires=[i, (i + 2)])
    qml.RY(-QP3[para_index], wires=6)
    # endregion

    # region 全连接层
    para_index = 0
    for i in range(5,9):
        qml.RY(QF[para_index], wires=5)
        para_index = para_index + 1
    qml.CNOT(wires=[5, 6])
    qml.CNOT(wires=[6, 5])
    qml.CNOT(wires=[7, 5])
    qml.CNOT(wires=[8, 6])
    # endregion



    return qml.probs(wires=[5,6])


class ATQCNNLayer(tf.keras.layers.Layer):

    def __init__(self, start_qubits, final_qubits):
        super(ATQCNNLayer, self).__init__()
        self.kernel = None
        self.start_qubits = start_qubits
        self.final_qubits = final_qubits
        self.QC1 = tf.Variable(np.random.random(12))  # 42
        self.QP1 = tf.Variable(np.random.random(2))  # 12
        self.QC2 = tf.Variable(np.random.random(12))  # 36
        self.QP2 = tf.Variable(np.random.random(2))  # 10
        self.QC3 = tf.Variable(np.random.random(12))  # 36
        self.QP3 = tf.Variable(np.random.random(2))  # 10
        self.QF = tf.Variable(np.random.random(4))  # 1
        self.dev = qml.device("default.qubit", wires=self.start_qubits)
        self.qnode = qml.QNode(ATQCNNcircuit, self.dev, interface='tf')

    def call(self, inputs):
        out = []
        inputs = tf.squeeze(inputs)
        # print(inputs.shape)
        for i in range(len(inputs)):
            try:
                out_temp = self.qnode(inputs[i], self.QC1, self.QC2, self.QC3, self.QP1, self.QP2, self.QP3, self.QF)
                out_temp = tf.stack(out_temp, axis=0)
                out_temp = tf.nn.softmax(out_temp)
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
optimizer = tf.keras.optimizers.Adam(learning_rate=0.025)
if __name__ == "__main__":
    try:
        with tf.device('/device:GPU:1'):

            train_data, train_label, test_data, test_label = Tool.readdata_2classes_1234()
            MyModel = ATQCNNLayer(start_qubits=9, final_qubits=2)
            epoch = 30
            batch_size = 100
            total_step = int(len(train_data) / batch_size)

            print("开始训练")

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

                    with open("ATQCNN_parameters.txt", 'w') as file:
                        file.write(str(MyModel.trainable_variables))

                test_label_pred = MyModel(test_data)
                test_acc, test_loss = evaluate_acc_loss(y_pred=test_label_pred, y_true=test_label)
                test_accuracy_results.append(test_acc)
                test_loss_results.append(test_loss)

                print("Epoch {:03d},:Loss: {:.3f}, Accuracy: {:.3%}".format(e + 1, test_loss, test_acc))
                with open("ATQCNN_2Classes.txt", 'w') as file:

                    file.write(str(test_accuracy_results) + "\n")
                with open("ATQCNN_2Code_Loss.txt", 'w') as file:
                    file.write(str(test_loss_results) + "\n")
    except RuntimeError as e:
        print(e)
