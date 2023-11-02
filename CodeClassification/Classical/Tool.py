import numpy as np
import csv
np.random.seed(42)


def readdata_2classes_15():
    # 读取两类
    filter_name = "F:\\wqspython\\QCNN-V1\\QCNN-V1\\CodeClassification\\data\\feature_vectors_syscalls_frequency_5_Cat.csv"
    read = csv.reader(open(filter_name, 'r'))
    l = [0,0]
    data = []

    for i in read:
        if i[139] == '1.00E+00' and l[0]<1000:
            temp = i[0:11] + i[12:26] + i[27:29] + i[30:41] + i[42:51] + i[53:63]+i[64:67]+i[68:105]+i[106:115]+i[116:138]
            temp = list(map(float, temp))
            if sum(temp) != 0:
                data.append(temp + ['0'])
                l[0] = l[0]+1
        if i[139] == '5.00E+00' and l[1]<1000:
            temp = i[0:11] + i[12:26] + i[27:29] + i[30:41] + i[42:51] + i[53:63]+i[64:67]+i[68:105]+i[106:115]+i[116:138]
            temp = list(map(float, temp))
            if sum(temp) != 0:
                data.append(temp + ['1'])
                l[1] = l[1] + 1
    print(len(data))

    data = np.array(data)
    np.random.shuffle(data)
    data = data.astype(float)

    data_train = data[0:1600].copy()
    data_test = data[1600:2000].copy()

    train_data, train_label = np.hsplit(data_train, [128])
    test_data, test_label = np.hsplit(data_test, [128])

    # train_data, test_data = np.expand_dims(train_data, axis=1), np.expand_dims(test_data, axis=1)
    train_label = np.squeeze(train_label)
    test_label = np.squeeze(test_label)

    return train_data, train_label, test_data, test_label

def readdata_2classes_15_one_hot():
    # 读取两类
    filter_name = "F:\\wqspython\\QCNN-V1\\QCNN-V1\\CodeClassification\\data\\feature_vectors_syscalls_frequency_5_Cat.csv"
    read = csv.reader(open(filter_name, 'r'))
    l = [0,0]
    data = []

    for i in read:
        if i[139] == '1.00E+00' and l[0]<500:
            temp = i[0:11] + i[12:26] + i[27:29] + i[30:41] + i[42:51] + i[53:63]+i[64:67]+i[68:105]+i[106:115]+i[116:138]
            temp = list(map(float, temp))
            if sum(temp) != 0:
                data.append(temp + ['1','0'])
                l[0] = l[0]+1
        if i[139] == '5.00E+00' and l[1]<500:
            temp = i[0:11] + i[12:26] + i[27:29] + i[30:41] + i[42:51] + i[53:63]+i[64:67]+i[68:105]+i[106:115]+i[116:138]
            temp = list(map(float, temp))
            if sum(temp) != 0:
                data.append(temp + ['0','1'])
                l[1] = l[1] + 1
    print(len(data))

    data = np.array(data)
    np.random.shuffle(data)
    data = data.astype(float)

    data_train = data[0:800].copy()
    data_test = data[800:1000].copy()

    train_data, train_label = np.hsplit(data_train, [128])
    test_data, test_label = np.hsplit(data_test, [128])
    train_label = np.squeeze(train_label)
    test_label = np.squeeze(test_label)

    return train_data, train_label, test_data, test_label

def readdata_2classes_1234():
    # 读取4类
    filter_name = "F:\\wqspython\\QCNN-V1\\QCNN-V1\\CodeClassification\\data\\feature_vectors_syscalls_frequency_5_Cat.csv"
    read = csv.reader(open(filter_name, 'r'))
    # for rea in read:
    #     print(rea)
    data = []
    l=[0,0,0,0]

    for i in read:
        if i[139] == '1.00E+00' and l[0]<1000:
            temp = i[0:12] + i[13:22] + i[23:35] + i[36:71] + i[72:77] + i[78:94]+i[95:100]+i[101:127]+i[130:138]
            print(len(temp))
            temp = list(map(float, temp))
            if sum(temp) != 0:
                data.append(temp + ['0'])
                l[0] = l[0]+1
        if i[139] == '2.00E+00' and l[1]<1000:
            temp = i[0:12] + i[13:22] + i[23:35] + i[36:71] + i[72:77] + i[78:94]+i[95:100]+i[101:127]+i[130:138]
            temp = list(map(float, temp))
            if sum(temp) != 0:
                data.append(temp + ['1'])
                l[1] = l[1] + 1
        if i[139] == '3.00E+00' and l[2]<1000:
            temp = i[0:12] + i[13:22] + i[23:35] + i[36:71] + i[72:77] + i[78:94]+i[95:100]+i[101:127]+i[130:138]
            temp = list(map(float, temp))
            if sum(temp) != 0:
                data.append(temp + ['2'])
                l[2] = l[2] + 1
        if i[139] == '4.00E+00' and l[3]<1000:
            temp = i[0:12] + i[13:22] + i[23:35] + i[36:71] + i[72:77] + i[78:94]+i[95:100]+i[101:127]+i[130:138]
            temp = list(map(float, temp))
            if sum(temp) != 0:
                data.append(temp + ['3'])
                l[3] = l[3] + 1
    print(len(data))  # 9772
    # print(data[2000][128])

    # print(data[1104])
    data = np.array(data)
    np.random.shuffle(data)
    data = data.astype(float)
    total_num = len(data)

    data_train = data[0:3200].copy()
    data_test = data[3200:4000].copy()

    train_data, train_label = np.hsplit(data_train, [128])
    test_data, test_label = np.hsplit(data_test, [128])
    train_label = np.squeeze(train_label)
    test_label = np.squeeze(test_label)

    return train_data, train_label, test_data, test_label


def readdata_2classes_1_5_frequency():
    # 读取两类
    filter_name = "F:\\wqspython\\QCNN-V1\\QCNN-V1\\CodeClassification\\data\\feature_vectors_syscalls_frequency_5_Cat.csv"
    read = csv.reader(open(filter_name, 'r'))

    data = []

    for row in read:
        if row[139] == '1.00E+00':
            data.append(row[0:139])
        if row[139] == '5.00E+00':
            data.append(row[0:139])
    print(len(data))

    frequency = []
    for i in range(139):
        frequency.append(0)
    for row in data:
        for i in range(len(row)):
            if row[i] != "0.00E+00":
                frequency[i] = frequency[i] + 1
    print(frequency)
    for i in range(11):
        minmum = min(frequency)
        index = frequency.index(minmum)
        frequency.remove(minmum)
        print(index+i)
        # print(minmum)
    # print(frequency)

def readdata_2classes_1234_frequency():
    # 读取五类
    filter_name = "F:\\wqspython\\QCNN-V1\\QCNN-V1\\CodeClassification\\data\\feature_vectors_syscalls_frequency_5_Cat.csv"
    read = csv.reader(open(filter_name, 'r'))

    data = []

    for row in read:
        if row[139] != "5.00E+00":
            data.append(row[0:139])
    # print(len(data))

    frequency = []
    for i in range(139):
        frequency.append(0)
    for row in data:
        for i in range(len(row)):
            if row[i] != "0.00E+00":
                frequency[i] = frequency[i] + 1
    # print(frequency)
    for i in range(11):
        minmum = min(frequency)
        index = frequency.index(minmum)
        frequency.remove(minmum)
        print(index+i)
        # print(minmum)
    # print(frequency)


if __name__ == "__main__":
    readdata_2classes_1234()
