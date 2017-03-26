import numpy as np
import tensorflow as tf

from tensorflow_vgg_train import utils
from tensorflow_vgg_train.vgg import vgg19

img1 = utils.load_image("./test_data/tiger.jpeg")[:, :, :3]    # Con [:, :, :3] podemos ahora cargar imagenes png y jpg
img2 = utils.load_image("./test_data/dog2.png")[:, :, :3]
img3 = utils.load_image("./test_data/lobo.png")[:, :, :3]

batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))
batch3 = img3.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2, batch3), 0)

print(batch.shape)

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device('/cpu:0'):
    with tf.Session() as sess:
        images = tf.placeholder("float", [None, 224, 224, 3])
        feed_dict = {images: batch}

        # vgg = vgg19.Vgg19('weight/test-save-vgg19.npy')
        vgg = vgg19.Vgg19()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        red = tf.Variable(tf.random_normal([3, 1]), name='weight')
        prob, kernel = sess.run([vgg.prob, vgg.get_conv_filter('conv1_1')], feed_dict=feed_dict)

        # kernel = vgg.data_dict['conv1_1']

        utils.print_prob(prob[0], './synset.txt')
        utils.print_prob(prob[1], './synset.txt')
        utils.print_prob(prob[2], './synset.txt')

        # print(kernel, kernel.shape)
        # print()
        # print(conv[0,:,:,3])
        # print()
        # print(conv[0,:,:,3].shape)

        # skimage.io.imshow(conv[0,:,:,1])
        # skimage.io.show()
        # skimage.io.imshow(conv[0, :, :, 2])
        # skimage.io.show()
        # skimage.io.imshow(conv[0, :, :, 3])
        # skimage.io.show()
        # skimage.io.imshow(conv[0, :, :, 4])
        # skimage.io.show()
        # print(red.get_shape().as_list())

