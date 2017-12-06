from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import json
import numpy as np

import chainer
from chainer.dataset.convert import concat_examples
from chainer import serializers

import nets

from data import get_multi_mnist_dataset 
from data import get_mnist_dataset
from data import fetch_new_batch


def save_images(xs, filename, marked_row=0):
    width = xs[0].shape[0]
    height = len(xs)

    xs = [np.array(x.tolist(), np.float32) for x in xs]
    # subplots with many figs are very slow
    fig, ax = plt.subplots(
        height, width, figsize=(1 * width / 2.5, height / 2.5))
    xs = np.concatenate(xs, axis=0)
    for i, (ai, xi) in enumerate(zip(ax.ravel(), xs)):
        ai.set_xticklabels([])
        ai.set_yticklabels([])
        ai.set_axis_off()
        color = 'Greens_r' if i // width == marked_row else 'Blues_r'
        ai.imshow(xi.reshape(36, 36), cmap=color, vmin=0., vmax=1.)

    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05)
    # saving and clearing subplots with many figs are also very slow
    fig.savefig(filename, bbox_inches='tight', pad=0.)
    plt.clf()
    plt.close('all')


def visualize_reconstruction(model, x, t, filename='vis.png'):
    print('visualize', filename)
    vs_norm, vs = model.output(x)
    x_recon = model.reconstruct(vs, t)
    save_images([x, x_recon.data],
                filename)


def visualize_reconstruction_alldigits(model, x, t, filename='vis_all.png'):
    print('visualize', filename)
    x_recon_list = []
    vs_norm, vs = model.output(x)
    for i in range(10):
        pseudo_t = model.xp.full(t.shape, i).astype('i')
        x_recon = model.reconstruct(vs, pseudo_t).data
        x_recon_list.append(x_recon)
    save_images([x] + x_recon_list,
                filename)


def visualize_reconstruction_tweaked(model, x, t, filename='vis_tweaked.png'):
    print('visualize', filename)
    x_recon_list = []
    vs_norm, vs = model.output(x)
    vs = vs.data
    vs = model.xp.concatenate([vs] * 16, axis=0)
    t = model.xp.concatenate([t] * 16, axis=0)
    I = model.xp.arange(16)
    for i in range(9):
        tweaked_vs = model.xp.array(vs)
        tweaked_vs[I, I, :] += (i - 4.) * 0.075  # raw + [-0.30, 0.30]
        x_recon = model.reconstruct(tweaked_vs, t).data
        x_recon_list.append(x_recon)
    x_recon = model.reconstruct(vs, t).data
    save_images(x_recon_list,
                filename,
                marked_row=4)

def get_mmnist_samples(dataset):
    # 2 samples for each digit
    samples = []
    while len(samples) < 20:
        batch = dataset.next()

        for i, (x, t) in enumerate(batch):
            if np.argmax(t[:,1]) == len(samples) // 2:
                print('{}-th sample is used'.format(i))
                samples.append((x, t))
            if len(samples) >= 20:
                break
    return samples

def get_samples(dataset):
    # 2 samples for each digit
    samples = []
    for i, (x, t) in enumerate(dataset):
        if t == len(samples) // 2:
            print('{}-th sample is used'.format(i))
            samples.append((x, t))
        if len(samples) >= 20:
            break
    return samples


if __name__ == '__main__':
    DATA_PATH = '/home/ccc2174/capsnet-segmentation/mmnist_data/'

    parser = argparse.ArgumentParser(
        description='CapsNet: MNIST reconstruction')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--load')
    parser.add_argument('--mmnist', '-m', action='store_true')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    model = nets.CapsNet(use_reconstruction=True, mmnist=args.mmnist)
    serializers.load_npz(args.load, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    if args.mmnist:
        _, test = get_multi_mnist_dataset(256, 100, path=DATA_PATH)
    else:
        _, test = get_mnist_dataset(256, 100)

    if args.mmnist:
        batch = get_mmnist_samples(test)
    else:
        batch = get_samples(test)
    x, t = fetch_new_batch(batch, args.gpu)

    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
           
            x_composed, x_a, x_b = np.split(x,indices_or_sections=3,axis=1)
            y_composed, y_a, y_b = np.split(t,indices_or_sections=3,axis=-1)

            y_composed = np.squeeze(y_composed, axis=(2,))
            
            visualize_reconstruction(model, x_composed, y_a, filename='vis_a.png')
            visualize_reconstruction_alldigits(model, x_composed, y_a, filename='vis_all_a.png')

            visualize_reconstruction(model, x_composed, y_b, filename='vis_b.png')
            visualize_reconstruction_alldigits(model, x_composed, y_b, filename='vis_all_b.png')
            for i in range(10):
                visualize_reconstruction_tweaked(
                    model, x_composed[i * 2: i * 2 + 1], y_a[i * 2: i * 2 + 1],
                    filename='vis_tweaked_a{}.png'.format(i))

            for i in range(10):
                visualize_reconstruction_tweaked(
                    model, x[i * 2: i * 2 + 1], t[i * 2: i * 2 + 1],
                    filename='vis_tweaked_b{}.png'.format(i))
