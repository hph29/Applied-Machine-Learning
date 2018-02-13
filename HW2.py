import numpy as np
import matplotlib.pyplot as plt
#https://maviccprp.github.io/a-support-vector-machine-in-just-a-few-lines-of-python-code/


def HW2():

    training_data, validation_data, testing_data = load_data()

    svm_sgd(training_data, validation_data, testing_data)


def load_data():
    # load training data set
    data = np.genfromtxt(fname="adult.data", delimiter=',')

    # remove non-continuous attribute
    data_x = np.ma.compress_cols(np.ma.masked_invalid(data))

    # rescale feature to normalize variance
    data_x = rescale_feature(data_x)

    # get label
    data_y = generate_label(path="adult.data")

    # combine feature and label for splitting
    combine = np.hstack((data_x, np.transpose(np.array([data_y]))))

    # split data set to 80% training, 10% validation, 10% testing
    num_row = np.shape(data_x)[0]
    training_index = round(num_row * 0.8)
    validation_index = round(num_row * 0.9)

    training_data = combine[0:training_index, :]

    validation_data = combine[training_index + 1: validation_index, :]

    testing_data = combine[validation_index + 1:, :]

    return training_data, validation_data, testing_data


def rescale_feature(x):
    for i in range(np.shape(x)[1]):
        column = x[:, i]
        std = np.std(column)
        mean = np.mean(column)
        for j in range(len(column)):
            column[j] = (column[j] - mean) / std
        x[:, i] = column
    return x


def generate_label(path):
    label = []
    with open(path) as csv_file:
        for line in csv_file:
            line = line.split(",")[-1].strip()
            if line:
                if line == '<=50K':
                    label.append(-1)
                elif line == '>50K':
                    label.append(1)
                else:
                    raise ValueError("Label cannot be " + line)
    return np.array(label)


def svm_sgd(train_data, validation_data, testing_data):
    steps = 300
    lambds = [0.0001, 0.001, 0.01, 0.1]
    train_data_partition = shuffle_and_partition(train_data, steps)

    epochs = 50

    accuracy = []
    weight = []
    for z in range(len(lambds)):
        lambd = lambds[z]
        w = np.zeros(len(train_data_partition[0][0, :])-1)
        b = 0
        for epoch in range(epochs):
            learning_rate = 1 / (0.01 * epoch + 50)
            for i in range(steps):
                # training_set
                batch = train_data_partition[epoch]
                np.random.shuffle(batch)
                x, y = get_feature_and_label(batch)
                t_x = x
                t_y = y
                # validation_set
                np.random.shuffle(validation_data)
                x, y = get_feature_and_label(validation_data)
                v_x = np.array([x[i, :] for i in range(0, 300, 6)])
                v_y = [y[i] for i in range(0, 300, 6)]

                if (t_y[i] * np.dot(t_x[i], w)) < 1:
                    w = w - learning_rate * ((lambd * w) - (t_x[i] * t_y[i]))
                    b = b + learning_rate * t_y[i]
                else:
                    w = w - learning_rate * (lambd * w)
                if (i + 1) % 30 == 0:
                    accuracy.append(predict(v_x, v_y, w, b))
                    weight.append(np.sqrt(w.dot(w)))
        test_x, test_y = get_feature_and_label(testing_data)
        print("Prediction based on λ = %.4f" % lambd + " is " + str(predict(test_x, test_y, w, b)))
    plot(accuracy, weight)


def plot(accuracy, weight):
    epoch = np.arange(0, 50, 0.1)
    accuracy = np.array(accuracy)
    weight = np.array(weight)

    fig = plt.figure(num=1, figsize=(13, 13), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(311)
    ax.plot(epoch, accuracy[0:500], label='λ = 1e-4')
    ax.plot(epoch, accuracy[500: 1000], label='λ = 1e-3')
    ax.plot(epoch, accuracy[1000: 1500], label='λ = 1e-2')
    ax.plot(epoch, accuracy[1500: 2000], label='λ = 1e-1')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='lower right', bbox_to_anchor=(1.12 ,-0.02))
    fig.savefig('samplefigure', bbox_extra_artists=(lgd,), bbox_inches='tight')

    fig2 = plt.figure(num=1, figsize=(13, 13), dpi=80, facecolor='w', edgecolor='k')
    ax2 = fig2.add_subplot(313)
    ax2.plot(epoch, weight[0:500], label='λ = 1e-4')
    ax2.plot(epoch, weight[500: 1000], label='λ = 1e-3')
    ax2.plot(epoch, weight[1000: 1500], label='λ = 1e-2')
    ax2.plot(epoch, weight[1500: 2000], label='λ = 1e-1')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Magnitude of w")
    handles2, labels2 = ax2.get_legend_handles_labels()
    lgd2 = ax2.legend(handles2, labels2, loc='lower right', bbox_to_anchor=(1.12 ,-0.02))
    fig2.savefig('samplefigure2', bbox_extra_artists=(lgd2,), bbox_inches='tight')
    fig2.subplots_adjust(bottom=0.2)

    plt.show()


def shuffle_and_partition(data, batch_size):
    np.random.shuffle(data)
    return [data[i * batch_size: (i + 1) * batch_size, :] for i in range(int(len(data)/batch_size))]


def get_feature_and_label(data):
    return data[:, 0:-1], data[:, -1]


def predict(x, y, w, b):
    prediction = []
    for i, j in enumerate(x):
        if np.sign(np.dot(w, x[i,]) + b) == y[i]:
            prediction.append(1)
        else:
            prediction.append(0)
    return np.sum(np.array(prediction)) / len(prediction)


if __name__ == "__main__":

    HW2()






