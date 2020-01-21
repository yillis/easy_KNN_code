# an easy KNN code

import numpy as np
from os import listdir


# the first two function is an easy example to understand KNN
# you init four point and each point correspond a label
# then you can use KNN to estimate the label of other points
def create_data_set():
    # each group correspond a label
    res_groups = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    res_labels = ['A', 'A ', 'B', 'B']
    return res_groups, res_labels


def get_in_data():
    # the point you want to estimate
    return [1, 1]


# in next two function, you use the 'digits' training data set to estimate
# have more data and KNN can have higher accuracy
def get_in_data_from_file(filename: str) -> np.ndarray:
    # from folder to get data
    res = np.zeros((1, 1024))
    with open(filename, 'r') as fp:
        for i in range(32):
            chs = fp.readline()
            for j in range(32):
                res[0, i * 32 + j] = int(chs[j])
    return res


def get_data_set(path: str) -> np.ndarray:
    # create data set
    dirs = listdir(path)
    m = len(dirs)
    res = np.zeros((m, 1024))
    labels = []
    for i in range(m):
        res[i] = get_in_data_from_file(path + '/' + dirs[i])[0]
        labels.append(dirs[i][0])
    return res, labels


# the main progress of KNN
def knn(in_data, data_set, labels, k):
    # deal the input data, get the same shape mat with data_set
    data_set_size = data_set.shape[0]
    mat = np.tile(in_data, (data_set_size, 1))

    # calculate the distance between input data and the data_set
    dis_power_mat = (mat - data_set) ** 2
    dis_mat = dis_power_mat.sum(len(data_set.shape) - 1) ** 0.5

    # count the number of occurrences of the k labels, and sort
    sorted_dis_index = dis_mat.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dis_index[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=lambda x: x[1], reverse=True)

    # return the maximum occurrence of the k labels
    return sorted_class_count[0][0]


if __name__ == '__main__':
    # you can use the first example like this
    # data_set, labels = create_data_set()
    # print(get_in_data(), data_set, labels, 1)

    # second example
    # you can change the value of k to observe the change in accuracy
    data_set, labels = get_data_set('digits/trainingDigits')
    dirs = listdir('digits/testDigits')
    cnt = 0
    m = len(dirs)
    print('there are %d test data' % m)
    for i in range(m):
        num = int(dirs[i][0])
        res = int(knn(get_in_data_from_file('digits/testDigits/%s' % dirs[i]),
                      data_set, labels, 4))
        if res == num:
            cnt += 1
        else:
            print('''it's not correct in %s, output %d.''' % (dirs[i], res))
    print('%d test data correctly, the accuracy is %f%%' % (cnt, cnt / m * 100))
