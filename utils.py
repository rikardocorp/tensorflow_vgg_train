import skimage
import skimage.io
import skimage.transform
import numpy as np
from datetime import datetime
import sys, os
import itertools
import matplotlib.pyplot as plt
from pathlib import Path

# synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path, scale=255, xrange=[0, 1]):
    # load image
    img = skimage.io.imread(path)
    img = img / scale
    assert (xrange[0] <= img).all() and (img <= xrange[1]).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


def save_image(path_source, path_dest, name_image, transform=False, path_csv=None):

    name, ext = name_image.split('.')

    f = open(path_csv, "a+")

    if transform is False:

        img = load_image(path_source+name_image)
        skimage.io.imsave(path_dest + name + '.' + ext, img)
        f.write(",".join(map(str, [name, 0])) + "\n")

    else:
        img = load_image(path_source + name_image)
        img90 = skimage.transform.rotate(img, 90, resize=True)
        img180 = skimage.transform.rotate(img, 180, resize=True)
        img270 = skimage.transform.rotate(img, 270, resize=True)

        skimage.io.imsave(path_dest + name+'_0.'+ext, img)
        f.write(",".join(map(str, [name+'_0', 1])) + "\n")
        skimage.io.imsave(path_dest + name+'_90.'+ext, img90)
        f.write(",".join(map(str, [name+'_90', 1])) + "\n")
        skimage.io.imsave(path_dest + name+'_180.'+ext, img180)
        f.write(",".join(map(str, [name+'_180', 1])) + "\n")
        skimage.io.imsave(path_dest + name+'_270.'+ext, img270)
        f.write(",".join(map(str, [name+'_270', 1])) + "\n")

    f.close()
    print('Save image: ', name_image)


def save_image_flip(path_source, path_dest, name_image, path_csv=None):
    name, ext = name_image.split('.')
    f = open(path_csv, "a+")

    img = load_image(path_source + name_image)
    imgflip = np.fliplr(img)
    skimage.io.imsave(path_dest + name+'_flip.'+ext, imgflip)
    f.write(",".join(map(str, [name+'_flip', 1])) + "\n")

    f.close()
    print('Save image: ', name_image)


# returns the top1 string
def print_prob(prob, file_path, top=5):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    # np.argsort -> ordena el array y almacena los INDICES de los numeros ordenados
    # x[::-1] -> invierte el orden de la lista 'x'
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    if top > 0:
        top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(top)]
        print(("Top"+str(top)+": ", top5))

    return top1


def print_prob_all(prob, file_path, top=5):
    synset = [l.strip() for l in open(file_path).readlines()]
    for i in range(len(prob)):
        _prob = prob[i]
        pred = np.argsort(_prob)[::-1]
        top1 = synset[pred[0]]
        print("    Top1: ", top1, _prob[pred[0]])

        if top > 0:
            topn = [(synset[pred[i]], _prob[pred[i]]) for i in range(top)]
            print("    Top" + str(top) + ": ", topn)


def print_accuracy(target, prob, matrix_confusion=None, predicted=[]):

    total = len(target)
    count = 0

    for i in range(total):
        true_result = np.argsort(prob[i])[::-1][0]
        if target[i] == true_result:
            count += 1

        predicted.append(true_result)
        matrix_confusion[target[i]][true_result] = matrix_confusion[target[i]][true_result] + 1

    accuracy = count / total
    print('    results[ Total:'+str(total)+' | True:'+str(count)+' | False:'+str(total-count)+' | Accuracy:'+str(accuracy)+' ]')
    return count, matrix_confusion, predicted


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def time_epoch(millis):
    millis = int(millis)
    seconds = (millis/1000) % 60
    seconds = int(seconds)
    minutes = (millis/(1000*60)) % 60
    minutes = int(minutes)
    hours = (millis/(1000*60*60)) % 24

    return hours, minutes, seconds


def test():
    img = skimage.io.imread("./test_data/tiger.jpeg")[:, :, :3]
    ny = 300
    nx = int(img.shape[1] * ny / img.shape[0])
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("./test_data/output.jpg", img)


def write_log(total_data, epoch, m_batch, l_rate, accuracy=0, file_npy='None'):
    now = datetime.now()
    id = int(now.timestamp()*1000000)
    date = now.strftime('%d-%m-%Y %H:%M:%S')
    file = sys.argv[0].split('/')[-1]

    f = open("log-server.txt", "a+")
    f.write('id:{}  date:{}  file:{}  input:{}  epoch:{}  m-batch:{}  l-rate:{}  accuracy:{:3.3f}  file_npy:{}\n'.format(id,date,file,total_data,epoch,m_batch,l_rate,accuracy,file_npy))
    f.close()
    print('Create log in log-server.txt:', id)


def save_layer_output(output, label, name="layer"):
    total = len(label)
    lab = np.reshape(label, (total, 1))
    res = np.concatenate((output, lab), axis=1)

    f = open("features/output_"+name+".csv", "a+")
    for i in range(total):
        f.write(",".join(map(str, res[i])) + "\n")
    f.close()
    print("    Save feature extractor, "+name)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def directory_exist(pathname):
    directory = Path(pathname)
    if not directory.exists():
        os.makedirs(pathname)
        print('El directorio {0} ha sido creado.'.format(pathname))
    else:
        print('El directorio {0} existe.'.format(pathname))


if __name__ == "__main__":
    test()
