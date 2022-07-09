#!/usr/bin/env python
# coding: utf-8

import os
import argparse

import tensorflow as tf
from tensorflow import keras


def tf_to_tflite(source1, source2):

    # Load Tensorflow model
    model = keras.models.load_model(f"{source1}.h5")
    model.load_weights(f"{source2}.hdf5")

    # Convert to TF-Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Write to .tflite file
    with open(f"{source1}.tflite", "wb") as f_out:
        f_out.write(tflite_model)


if __name__ == "__main__":

    # Initialize arguments parser
    def file_choices(choices, fname):
        ext = os.path.splitext(fname)[1][1:]
        if ext not in choices:
            parser.error("file doesn't end with one of {}".format(choices))
        return os.path.splitext(fname)[0]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "h5",
        type=lambda s: file_choices(("h5"), s),
        help="The path to the .h5 file (model)",
    )
    parser.add_argument(
        "hdf5",
        type=lambda s: file_choices(("hdf5"), s),
        help="The path to the .hdf5 file (weights)",
    )
    args = parser.parse_args()

    # Train / Evaluate / Save
    print(">>> LET'S CONVERT A NEW TF-LITE MODEL")
    tf_to_tflite(args.h5, args.hdf5)
    print(">>> NEW MODEL AVAILABLE")
