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
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
if __name__ == "__main__":
    try:
        with tf.device('/device:GPU:1'):

            train_data, train_label, test_data, test_label = Tool.readdata_2classes_15()
            print(train_data.shape)
            input_shape = (50, 128, 1)
            MyModel = tf.keras.Sequential([
                tf.keras.layers.Conv1D(8, 3, input_shape=input_shape[1:],activation='relu'),
                tf.keras.layers.MaxPool1D(2),
                tf.keras.layers.Conv1D(8, 3, activation='relu'),
                tf.keras.layers.MaxPool1D(2),
                tf.keras.layers.Conv1D(8, 3, activation='relu'),
                tf.keras.layers.MaxPool1D(2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax)
            ])
            print(train_data[0:10].shape)
            MyModel(train_data[0:10])
            MyModel.summary()
            epoch = 40
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


                test_label_pred = MyModel(test_data)
                test_acc, test_loss = evaluate_acc_loss(y_pred=test_label_pred, y_true=test_label)
                test_accuracy_results.append(test_acc)
                test_loss_results.append(test_loss)

                print("Epoch {:03d},:Loss: {:.3f}, Accuracy: {:.3%}".format(e + 1, test_loss, test_acc))
                with open("Classical_2Code_Acc.txt", 'w') as file:

                    file.write(str(test_accuracy_results) + "\n")
                with open("Classical_2Code_Loss.txt", 'w') as file:
                    file.write(str(test_loss_results) + "\n")
    except RuntimeError as e:
        print(e)
