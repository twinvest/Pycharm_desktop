import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical


def load_data(target_class=[3, 5], rescale=True, to_gray=True, use_validation=False):
    if to_gray:
        gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def pre_processing(x, y):
        keep = np.where(np.isin(y, target_class))
        x = x[keep[0]]
        y = y[keep[0]]
        for i, cls in enumerate(target_class):
            y[np.where(y == cls)] = i
        y = to_categorical(y, len(target_class))
        if to_gray:
            x = gray(x)
            x = np.expand_dims(x, -1)
        if rescale:
            x = x.astype('float32') / 255
        return x, y

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train, y_train = pre_processing(x_train, y_train)
    x_test, y_test = pre_processing(x_test, y_test)

    # data normalization
    #mean = np.mean(x_train, axis=(0, 1, 2, 3))
    #std = np.std(x_train, axis=(0, 1, 2, 3))
    #x_train = (x_train - mean) / (std + 1e-7)
    #x_test = (x_test - mean) / (std + 1e-7)

    if not use_validation:
        print(' x_train.shape :', x_train.shape)  # (10000, 32, 32, 1 or 3)
        print(' y_train.shape :', y_train.shape)  # (10000, 2)
        print(' x_test.shape :', x_test.shape)  # (2000, 32, 32, 1 or 3)
        print(' y_test.shape :', y_test.shape)  # (2000, 2)
        return (x_train, y_train), (x_test, y_test)
    else:
        x_val = x_train[8000:]
        y_val = y_train[8000:]
        x_train = x_train[:8000]
        y_train = y_train[:8000]
        print(' x_train.shape :', x_train.shape)  # (8000, 32, 32, 1 or 3)
        print(' y_train.shape :', y_train.shape)  # (8000, 2)
        print(' x_val.shape :', x_val.shape)  # (2000, 32, 32, 1 or 3)
        print(' y_val.shape :', y_val.shape)  # (2000, 2)
        print(' x_test.shape :', x_test.shape)  # (2000, 32, 32, 1 or 3)
        print(' y_test.shape :', y_test.shape)  # (2000, 2)
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def draw_result(logs, use_validation, file_path='./result.png'):
    if use_validation:
        test_label = 'Val'
    else:
        test_label = 'Test'

    y_vloss = logs['val_loss']
    y_loss = logs['loss']
    y_vacc = logs['val_acc']
    y_acc = logs['acc']

    x_len = np.arange(len(y_loss))
    fig = plt.figure(figsize=(12, 5))
    title = ''
    if 'hypers' in logs:
        title = ''
        for k, v in logs['hypers'].items():
            if k == 'optimizer':
                title += k + ':' + str(v.__class__.__name__) + '   '
            else:
                title += k + ':' + str(v) + '   '
    title += 'Test_Acc:{}%'.format(str(logs['test_acc']*100)[:5])
    fig.suptitle(title)#, fontsize="x-large")

    plt.rcParams['figure.constrained_layout.use'] = True

    plt.subplot(121)
    plt.title('Train/'+test_label+' Loss')
    plt.plot(x_len, y_vloss, marker='.', c='red', label=test_label + "-set Loss")
    plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")
    plt.xlim(left=-1)
    plt.xticks(x_len.tolist(), (x_len+1).tolist())
    plt.ylim(bottom=0, top=1.5)
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    idx = np.argmin(y_vloss)
    v = np.min(y_vloss)
    plt.annotate('min:'+str(v)[:5], xy=(idx, v), xytext=(idx, v + 0.15),
                 arrowprops=dict(arrowstyle="->", facecolor='black'), fontsize=10,
                 )

    plt.subplot(122)
    plt.title('Train/' + test_label + ' Accuracy')
    plt.plot(x_len, y_vacc, marker='.', c='red', label=test_label + "-set Accuracy")
    plt.plot(x_len, y_acc, marker='.', c='blue', label="Train-set Accuracy")
    plt.xlim(left=-1)
    plt.xticks(x_len.tolist(), (x_len+1).tolist())
    plt.ylim(bottom=0, top=1)
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')

    idx = np.argmax(y_vacc)
    v = np.max(y_vacc)
    plt.annotate('max:'+str(v*100)[:5]+'%', xy=(idx, v), xytext=(idx, v + 0.1),
                 arrowprops=dict(arrowstyle="->", facecolor='black'), fontsize=10,
                 )

    plt.savefig(fname=file_path)
    plt.clf()

c2n = {"Cat":0, "Dog":1}
n2c = {0:"Cat", 1:"Dog"}

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data([3,5], to_gray=True)
    index = 0

    img = np.squeeze(x_train[index])
    plt.imshow(img, cmap=plt.get_cmap(name='gray'))
    plt.text(14, -2, n2c[np.argmax(y_train[index])], fontsize=12)
    plt.show()