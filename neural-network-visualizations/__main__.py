#!/usr/bin/env python3
import argparse
from math import ceil, sqrt
from pathlib import Path
import time

import imageio
from keras import backend as K
from keras.applications import vgg16
from keras.layers import Conv2D
from keras.preprocessing import image
import numpy as np
import scipy

# Parameters for artificially generated inputs
input_width = 128
input_height = 128


def parse_args():
    parser = argparse.ArgumentParser(description='Neural network filter visualization.')
    parser.add_argument('input', nargs='?', help='input image for the network')
    return parser.parse_args()


def to_image(x):
    # Convert to RGB
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize_to_image(x):
    # Normalize to [0, 1]
    x_min = x.min()
    x = (x - x_min) / (x.max() - x_min + K.epsilon())
    return to_image(x)


def center_to_image(x):
    # Center on 0.5 with std 0.1
    x -= x.mean()
    x = 0.1 * x / (x.std() + K.epsilon()) + .5
    return to_image(x)


def stitch(images, margin=8):
    border_image_count = ceil(sqrt(len(images)))
    shape = images[0].shape
    if len(shape) == 2:
        image_height, image_width = shape
        channel_count = 1
    else:
        image_height, image_width, channel_count = shape
    width = border_image_count * (image_width + margin) - margin
    height = border_image_count * (image_height + margin) - margin
    if channel_count > 1:
        stitched = np.zeros((height, width, channel_count), dtype='uint8')
    else:
        stitched = np.zeros((height, width), dtype='uint8')

    # Stitch into the large image
    images = iter(images)
    for i in range(border_image_count):
        for j in range(border_image_count):
            image = next(images, None)
            if image is None:
                break
            x = (image_width + margin) * j
            y = (image_height + margin) * i
            if channel_count > 1:
                stitched[y:y + image_height, x:x + image_width, :] = image
            else:
                stitched[y:y + image_height, x:x + image_width] = image
    return stitched


def normalize(x):
    # Normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def save_kernels(model):
    for layer in model.layers:
        if not isinstance(layer, Conv2D):
            continue
        print('Saving kernels of layer {}'.format(layer.name))

        # Get input weights, ignoring biases
        weights = layer.get_weights()[0]
        # Format 3x3 kernels and save no more than 25
        weights = weights.transpose(2, 3, 0, 1).reshape((-1, 3, 3))[:25]
        weights = [normalize_to_image(scipy.misc.imresize(x, 16., interp='nearest')) for x in weights]
        imageio.imwrite('kernels_{}.png'.format(layer.name), stitch(weights, margin=2))


def generate_activations(model):
    '''Based on https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html'''
    # Placeholder for the input images
    input_img = model.input
    # Step size for gradient ascent
    step = 1.
    for layer in model.layers:
        if not isinstance(layer, Conv2D):
            continue
        print('Generating activations of layer {}'.format(layer.name))

        images_loss = []
        for filter_index in range(min(layer.output_shape[-1], 10)):
            print('Processing filter {} of {}'.format(filter_index, layer.name))

            # Build a loss function that maximizes the activation of the nth filter of the layer
            loss = K.mean(layer.output[:, :, :, filter_index])

            # Compute the gradient of the input picture with respect to this loss
            grads = K.gradients(loss, input_img)[0]
            # Normalize the gradient
            grads = normalize(grads)
            # Function that returns the loss and gradient for an input picture
            iterate = K.function([input_img], [loss, grads])

            # Start from a gray image with random noise
            input_image = np.random.random((1, input_width, input_height, 3))
            input_image = (input_image - 0.5) * 20 + 128

            # Run gradient ascent
            for i in range(80):
                loss_value, grads_value = iterate([input_image])
                input_image += grads_value * step

                print('Current loss value:', loss_value)
                if loss_value <= 0.:
                    # Some filters get stuck in 0
                    break

            if loss_value > 0:
                images_loss.append((loss_value, center_to_image(input_image[0])))

        if images_loss:
            # Take the 64 images with the highest loss
            images_loss = sorted(images_loss, reverse=True)[:64]
            images_loss = [x[1] for x in images_loss]
            imageio.imwrite('filter_inputs_{}.png'.format(layer.name), stitch(images_loss))


def get_layer_n_output(model, n, image):
    f = K.function([model.layers[0].input], [model.layers[n].output])
    return f([image])[0]


def show_features(model, image_path):
    batch = np.expand_dims(image.img_to_array(image.load_img(image_path)), axis=0)
    batch = vgg16.preprocess_input(batch)
    for i, layer in enumerate(model.layers):
        if not isinstance(layer, Conv2D):
            continue
        print('Generating features for layer {}'.format(layer.name))
        features = get_layer_n_output(model, i, batch)[0]
        features = [normalize_to_image(x) for x in features.transpose(2, 0, 1)[:64]]
        imageio.imwrite('features_{}_{}.png'.format(Path(image_path).stem, layer.name), stitch(features))


def main(args):
    if K.image_data_format() != 'channels_last':
        raise RuntimeError('This program requires data_format == "channels_last"')

    model = vgg16.VGG16(weights='imagenet', include_top=False)
    model.summary()

    if args.input:
        show_features(model, args.input)
    else:
        save_kernels(model)
        generate_activations(model)

if __name__ == '__main__':
    main(parse_args())
