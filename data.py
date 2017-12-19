from __future__ import print_function

import numpy as np

import os, sys

import chainer
from chainer.dataset import concat_examples
from chainer.datasets import tuple_dataset


def get_multi_mnist_dataset(batchsize, testsize, path='./'):
    try:
        raw = np.load(os.path.join(path, 'train.npz'))
        train = tuple_dataset.TupleDataset(raw['x'][:100], raw['y'][:100]) 

        raw = np.load(os.path.join(path, 'test.npz'))
        test = tuple_dataset.TupleDataset(raw['x'][:100], raw['y'][:100]) 
    except:
        print(sys.exc_info())
        print('failed to find data in path: ' + path)
        train, test = chainer.datasets.get_mnist(ndim=3)
        train = _combine_images(train, path, save='train', num_samples=60000)
        test = _combine_images(test, path, save='test', num_samples=10000)

    return _generate_iterator(train, test, batchsize, testsize)


def get_mnist_dataset(batchsize, testsize):
    train, test = chainer.datasets.get_mnist(ndim=3)

    return _generate_iterator(train, test, batchsize, testsize)

   
def fetch_new_batch(batch, gpu):
        m = concat_examples(batch, gpu)
        if type(m) == type({}):
            x, t = m['x'], m['y']
        else:
            x, t = m
            
        x = x.astype('float32')
        t = t.astype('int32')

        return x, t


def _generate_iterator(train, test, batchsize, testsize):
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, testsize,
                                                 repeat=False, shuffle=False)

    return train_iter, test_iter


def _shift_image(img):
    ''' shifts 28 x 28 image in both axes a maax of 4 pixels in
    each direction resulting in a 36x36 image '''

    new = np.zeros((1, 36,36))

    shift_r = np.random.randint(0,9,dtype='int')
    shift_d = np.random.randint(0,9,dtype='int')

    for x in range(28):
        for y in range(28):
            new[0, x+shift_r, y+shift_d] = img[0, x,y]

    return new


def _one_hot_y(y, num_classes=10):
    one_hot_y = np.zeros(num_classes)

    one_hot_y[y] = 1.

    return one_hot_y


def _combine_images(starting_images, path, save='train', num_samples=60000):
    count = 0 
    mm = []

    x = np.empty((num_samples,3,36,36), dtype=np.float32)
    y = np.empty((num_samples,10,3), dtype=np.uint8)

    while count < num_samples:
        idx_a = np.random.randint(0, num_samples)
        idx_b = np.random.randint(0, num_samples)

        img_a, num_a = starting_images[idx_a]
        img_b, num_b = starting_images[idx_b]
    
        if num_a != num_b:
            shift_img_a = _shift_image(img_a)
            shift_img_b = _shift_image(img_b)
            mm_img = np.logical_or(shift_img_a,shift_img_b, dtype='f')
            images = np.concatenate([mm_img,shift_img_a,shift_img_b], axis=0)

            hot_num_a, hot_num_b = _one_hot_y(num_a), _one_hot_y(num_b)
            mm_num = np.logical_or(hot_num_a, hot_num_b, dtype='f')
            nums = np.stack([mm_num, hot_num_a, hot_num_b], axis=-1)

            x[count] = images
            y[count] = nums

            count += 1
            if count % 1000 == 0:
                print(count, '/', num_samples)

    # save results
    np.savez_compressed(os.path.join(path, save), x=x, y=y)

    return tuple_dataset.TupleDataset(x, y) 
