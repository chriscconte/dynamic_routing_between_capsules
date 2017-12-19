from __future__ import print_function

import argparse
import json
import numpy as np

import chainer
from chainer.dataset import concat_examples
from chainer.datasets import tuple_dataset
from chainer import serializers
import nets
import cv2

from data import get_multi_mnist_dataset 
from data import get_mnist_dataset
from data import fetch_new_batch

DATA_PATH = './mmnist_data/'

def report(epoch, result, reconstruct):
    mode = 'train' if chainer.config.train else 'test '
    print('epoch {:2d}\t{} mean loss: {}, accuracy: {}'.format(
        epoch, mode, result['mean_loss'], result['accuracy']))
    if reconstruct:
        print('\t\t\tclassification: {}, reconstruction: {}'.format(
            result['cls_loss'], result['rcn_loss']))


def main():
    parser = argparse.ArgumentParser(description='CapsNet: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=256)
    parser.add_argument('--decay', '-d', type=float, default=0.95)
    parser.add_argument('--epoch', '-e', type=int, default=500)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--seed', '-s', type=int, default=789)
    parser.add_argument('--reconstruct', '--recon', action='store_true')
    parser.add_argument('--save', type=str, default='model')
    parser.add_argument('--mmnist', '-m', action='store_true')
    parser.add_argument('--load', type=str, default='')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    # Set up a neural network to train
    np.random.seed(args.seed)
    model = nets.CapsNet(use_reconstruction=args.reconstruct, mmnist=args.mmnist)

    # load model to continue training
    if args.load != '':
        serializers.load_npz(args.load, model)

    if args.gpu >= 0:
        # Make a speciied GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU
    np.random.seed(args.seed)
    model.xp.random.seed(args.seed)

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(alpha=1e-3)
    optimizer.setup(model)

    # Load the MNIST dataset or create MNIST 
    if args.mmnist:
        train_iter, test_iter = get_multi_mnist_dataset(args.batchsize, 100, path=DATA_PATH)
    else:
        train_iter, test_iter = get_mnist_dataset(args.batchsize, 100)

    best = 0.
    best_epoch = 0
    print('TRAINING starts')
    while train_iter.epoch < args.epoch:
        batch = train_iter.next()
        x, t = fetch_new_batch(batch, args.gpu)
        optimizer.update(model, x, t)

        # evaluation
        if train_iter.is_new_epoch:
            result = model.pop_results()
            report(train_iter.epoch, result, args.reconstruct)

            #with chainer.no_backprop_mode():
            #    with chainer.using_config('train', False):
            #        for batch in test_iter:
            #            x, t = fetch_new_batch(batch, args.gpu)
            #            
            #            x_comp, a, b = np.split(x,indices_or_sections=3,axis=1)
            #            y_composed, y_a, y_b = np.split(t,indices_or_sections=3,axis=-1)
            #            for i in range(10):
            #                cv2.imwrite(str(np.argmax(y_a[i]))+str(np.argmax(y_b[i]))+'.png', np.squeeze(x_comp[i])*255)
            #                cv2.waitKey(0)

            #            loss = model(x, t)

            #        result = model.pop_results()
            #        report(train_iter.epoch, result, args.reconstruct)

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
