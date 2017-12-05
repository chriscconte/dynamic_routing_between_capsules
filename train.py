from __future__ import print_function

import argparse
import json
import numpy as np

import chainer
from chainer.dataset import concat_examples
from chainer import serializers
import nets
import cv2


def shift_image(img):
    ''' shifts 28 x 28 image in both axes a maax of 4 pixels in
    each direction resulting in a 36x36 image '''

    new = np.zeros((1, 36,36))

    shift_r = np.random.randint(0,9,dtype='int')
    shift_d = np.random.randint(0,9,dtype='int')

    for x in range(28):
        for y in range(28):
            new[0, x+shift_r, y+shift_d] = img[0, x,y]

    return new

def one_hot_y(y, num_classes=10):
    one_hot_y = np.zeros(num_classes)

    one_hot_y[y] = 1.

    return one_hot_y


def combine_images(starting_images, save='train', num_samples=60000):

    count = 0
    mm = []
    num_samples = 2000
    for img_a, num_a in starting_images:
        for img_b, num_b in starting_images:
            if count >= num_samples:
                break
            if num_a != num_b:
                count += 1
                shift_img_a = shift_image(img_a)
                shift_img_b = shift_image(img_b)
                mm_img = np.logical_or(shift_img_a,shift_img_b, dtype='f')
                images = np.concatenate([mm_img,shift_img_a,shift_img_b], axis=0)


                hot_num_a, hot_num_b = one_hot_y(num_a), one_hot_y(num_b)
                mm_num = np.logical_or(hot_num_a, hot_num_b, dtype='f')
                nums = np.stack([mm_num, hot_num_a, hot_num_b], axis=-1)

                mm.append({'x': images, 'y':nums})

                if count % 1000 == 0:
                    print(count, '/', num_samples)

    # save results
    np.save('./mmnist_data/' + save, mm)

    return mm


def main():
    parser = argparse.ArgumentParser(description='CapsNet: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=1)
    parser.add_argument('--decay', '-d', type=float, default=0.95)
    parser.add_argument('--epoch', '-e', type=int, default=500)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--seed', '-s', type=int, default=789)
    parser.add_argument('--reconstruct', '--recon', action='store_true')
    parser.add_argument('--save', type=str, default='model')
    parser.add_argument('--mmnist', '-m', action='store_true')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    # Set up a neural network to train
    np.random.seed(args.seed)
    model = nets.CapsNet(use_reconstruction=args.reconstruct, mmnist=args.mmnist)
    if args.gpu >= 0:
        # Make a speciied GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU
    np.random.seed(args.seed)
    model.xp.random.seed(args.seed)

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(alpha=1e-3)
    optimizer.setup(model)

    # Load the MNIST dataset or create MMNIST 
    train, test = chainer.datasets.get_mnist(ndim=3)
    if args.mmnist:
        try:
            raise Exception('for now') 
            train = [tuple(i) for i in np.load('./mmnist_data/train.npy')]
            test = [tuple(i) for i in np.load('./mmnist_data/test.npy')]
        except:
            train = combine_images(train, save='train', num_samples=600000)
            test = combine_images(test, save='test', num_samples=10000)
        
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, 100,
                                                 repeat=False, shuffle=False)


    def report(epoch, result):
        mode = 'train' if chainer.config.train else 'test '
        print('epoch {:2d}\t{} mean loss: {}, accuracy: {}'.format(
            train_iter.epoch, mode, result['mean_loss'], result['accuracy']))
        if args.reconstruct:
            print('\t\t\tclassification: {}, reconstruction: {}'.format(
                result['cls_loss'], result['rcn_loss']))

    best = 0.
    best_epoch = 0
    print('TRAINING starts')
    while train_iter.epoch < args.epoch:
        batch = train_iter.next()
        m = concat_examples(batch, args.gpu)
        if type(m) == type({}):
            x, t = m['x'], m['y']
        else:
            x, t = m
            
        x = x.astype('float32')
        t = t.astype('int32')

        optimizer.update(model, x, t)

        # evaluation
        if True: #train_iter.is_new_epoch:
            result = model.pop_results()
            report(train_iter.epoch, result)

            with chainer.no_backprop_mode():
                with chainer.using_config('train', False):
                    for batch in test_iter:
                        m = concat_examples(batch, args.gpu)
                        if type(m) == type({}):
                            x, t = m['x'], m['y']
                        else:
                            x, t = m
                        x = x.astype('float32')
                        t = t.astype('int32')
                        loss = model(x, t)

                    result = model.pop_results()
                    report(train_iter.epoch, result)
            if train_iter.epoch % 10 == 0:
                serializers.save_npz(args.save+'_'+str(train_iter.epoch), model)
            if result['accuracy'] > best:
                best, best_epoch = result['accuracy'], train_iter.epoch
                serializers.save_npz(args.save, model)

            optimizer.alpha *= args.decay
            optimizer.alpha = max(optimizer.alpha, 1e-5)
            print('\t\t# optimizer alpha', optimizer.alpha)
            test_iter.reset()
    print('Finish: Best accuray: {} at {} epoch'.format(best, best_epoch))


if __name__ == '__main__':
    main()
