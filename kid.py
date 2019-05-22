from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import tempfile

from absl.testing import parameterized
import numpy as np
from scipy import linalg as scp_linalg

from google.protobuf import text_format
import tensorflow as tf

from tensorflow.contrib.gan.python.eval.python import classifier_metrics_impl as classifier_metrics
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

import functools
from PIL import Image
import PIL
import glob
from pathlib import Path

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_dir', type=str, default='results/258')
    parser.add_argument('--fake_dir', type=str, default='results/566')
    parser.add_argument('--get_fid', action=store_true)
    parser.add_argument('--get_sanity_check', action=store_true)
    parser.add_argument('--get_embeddings', action=store_true)

    args = parser.parse_args()

    # CHANGE THESE PATHS TO DIRS
    real_dir = Path(args.real_dir)
    fake_dir = Path(args.fake_dir)

    def load_image( image_path ):
        img = Image.open( image_path )
        newImg = img.resize((299,299), PIL.Image.BILINEAR).convert("RGB")
        data = np.array( newImg.getdata() )
        return 2*( data.reshape( (newImg.size[0], newImg.size[1], 3) ).astype( np.float32 )/255 ) - 1

    real_imgs = []
    for x in real_dir.iterdir():
        if 'png' in str(x):
            real_img = load_image(x)
            real_imgs.append(real_img)
    # print(real_imgs)

    fake_imgs = []
    for x in fake_dir.iterdir():
        if 'png' in str(x):
            fake_img = load_image(x)
            fake_imgs.append(fake_img)
    # print(fake_imgs)

    real_imgs = np.array(real_imgs)
    print(real_imgs.shape)
    fake_imgs = np.array(fake_imgs)
    print(fake_imgs.shape)

    # KID
    kid = classifier_metrics.kernel_inception_distance(real_images=real_imgs, generated_images=fake_imgs)
    with tf.Session() as sess:
        print('KID:', sess.run(kid))

    # Kernel Classifier Distance with Inception (technically KID)
    if args.get_sanity_check:
        kid_general = classifier_metrics.kernel_classifier_distance(real_images=real_imgs, generated_images=fake_imgs, classifier_fn=classifier_metrics.run_inception)
        with tf.Session() as sess:
            print('KID sanity check:', sess.run(kid_general))

    # FID
    if args.get_fid:
        fid = classifier_metrics.frechet_inception_distance(real_images=real_imgs, generated_images=fake_imgs)
        with tf.Session() as sess:
            print('FID:', sess.run(fid))

    # Get imagenet embeddings
    if args.get_embeddings:
        inception_embeddings_real = classifier_metrics.run_inception(real_imgs)
        inception_embeddings_fake = classifier_metrics.run_inception(fake_imgs)
        with tf.Session() as sess:
            # TODO: save embeddings
            print(sess.run(inception_embeddings_real))
            print(sess.run(inception_embeddings_fake))

if __name__ == '__main__':
    main()








