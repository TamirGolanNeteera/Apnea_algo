#from __future__ import print_function
import argparse
import sys
import os
conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append('/Neteera/Work/homes/dana.shavit/work/300622/Vital-Signs-Tracking/pythondev/')
sys.path.append(conf_path + '/Tests/NN')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

############################=========================================================
import tensorflow as tf

import keras
from keras import layers
#import tensorflow_addons as tfa

import numpy as np

import math



NUM_PATCHES = 180
#print("NUM_PATCHES", NUM_PATCHES)
# ViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 32
NUM_HEADS = 4
NUM_LAYERS = 4
MLP_UNITS = [
    PROJECTION_DIM * 2,
    PROJECTION_DIM,
]

# TOKENLEARNER
NUM_TOKENS = 8


def position_embedding(
    projected_patches, num_patches=NUM_PATCHES, projection_dim=PROJECTION_DIM
):
    # Build the positions.
    positions = tf.range(start=0, limit=num_patches, delta=1)

    # Encode the positions with an Embedding layer.
    encoded_positions = layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim
    )(positions)

    # Add encoded positions to the projected patches.
    print("projected_patches.shape",projected_patches.shape)
    print("encoded_positions.shape",encoded_positions.shape)
    return projected_patches + encoded_positions

# MLP block for Transformer
# This serves as the Fully Connected Feed Forward block for our Transformer.

def mlp(x, dropout_rate, hidden_units):
    # Iterate over the hidden units and
    # add Dense => Dropout.
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# TokenLearner module
# The following figure presents a pictorial overview of the module (source).
# TokenLearner module GIF
# The TokenLearner module takes as input an image-shaped tensor. It then passes it through multiple single-channel convolutional layers extracting different spatial attention maps focusing on different parts of the input. These attention maps are then element-wise multiplied to the input and result is aggregated with pooling. This pooled output can be trated as a summary of the input and has much lesser number of patches (8, for example) than the original one (196, for example).

# Using multiple convolution layers helps with expressivity. Imposing a form of spatial attention helps retain relevant information from the inputs. Both of these components are crucial to make TokenLearner work, especially when we are significantly reducing the number of patches.

def token_learner(inputs, number_of_tokens=NUM_TOKENS):
    # Layer normalize the inputs.
    x = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(inputs)  # (B, H, W, C)

    # Applying Conv2D => Reshape => Permute
    # The reshape and permute is done to help with the next steps of
    # multiplication and Global Average Pooling.
    attention_maps = keras.Sequential(
        [
            # 3 layers of conv with gelu activation as suggested
            # in the paper.
            layers.Conv2D(
                filters=number_of_tokens,
                kernel_size=(3, 3),
                activation=tf.nn.gelu,
                padding="same",
                use_bias=False,
            ),
            layers.Conv2D(
                filters=number_of_tokens,
                kernel_size=(3, 3),
                activation=tf.nn.gelu,
                padding="same",
                use_bias=False,
            ),
            layers.Conv2D(
                filters=number_of_tokens,
                kernel_size=(3, 3),
                activation=tf.nn.gelu,
                padding="same",
                use_bias=False,
            ),
            # This conv layer will generate the attention maps
            layers.Conv2D(
                filters=number_of_tokens,
                kernel_size=(3, 3),
                activation="sigmoid",  # Note sigmoid for [0, 1] output
                padding="same",
                use_bias=False,
            ),
            # Reshape and Permute
            layers.Reshape((-1, number_of_tokens)),  # (B, H*W, num_of_tokens)
            layers.Permute((2, 1)),
        ]
    )(
        x
    )  # (B, num_of_tokens, H*W)

    # Reshape the input to align it with the output of the conv block.
    num_filters = inputs.shape[-1]
    inputs = layers.Reshape((1, -1, num_filters))(inputs)  # inputs == (B, 1, H*W, C)

    # Element-Wise multiplication of the attention maps and the inputs
    attended_inputs = (
        attention_maps[..., tf.newaxis] * inputs
    )  # (B, num_tokens, H*W, C)

    # Global average pooling the element wise multiplication result.
    outputs = tf.reduce_mean(attended_inputs, axis=2)  # (B, num_tokens, C)
    return outputs

#Transformer block
def transformer(encoded_patches):
    # Layer normalization 1.
    print("encoded_patches.shape", encoded_patches.shape)
    x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(encoded_patches)
    print("x1.shape", x1.shape)
    # Multi Head Self Attention layer 1.
    attention_output = layers.MultiHeadAttention(
        num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
    )(x1, x1)

    # Skip connection 1.
    print("attention_output.shape", attention_output.shape)
    print("encoded_patches.shape", encoded_patches.shape)
    x2 = layers.Add()([attention_output, encoded_patches])

    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x2)

    # MLP layer 1.
    x4 = mlp(x3, hidden_units=MLP_UNITS, dropout_rate=0.1)

    # Skip connection 2.
    print("layers.Add()([attention_output, encoded_patches]).shape", x2.shape)
    print("mlp(x3, hidden_units=MLP_UNITS, dropout_rate=0.1).shape", x4.shape)
    encoded_patches = layers.Add()([x4, x2])
    print("encoded_patches = = layers.Add()([x4, x2]).shape", encoded_patches.shape)
    return encoded_patches

#ViT model with the TokenLearner module
def create_vit_1d(input_shape, use_token_learner=True, token_learner_units=NUM_TOKENS):
    inputs = layers.Input(shape=input_shape)  # (B, H, W, C)

    print(inputs.shape)

    projected_patches = layers.Conv1D(
        filters=PROJECTION_DIM,
        kernel_size=180,#kernel_size=50,
        strides=50,#strides=50,
        padding="SAME",
    )(inputs)
    print(projected_patches.shape)

    print(projected_patches.shape)
    h, w, c = projected_patches.shape
#    projected_patches = layers.Reshape((h * w, c))(
    projected_patches = layers.Reshape((w, c))(
        projected_patches
    )  # (B, number_patches, projection_dim)

    # Add positional embeddings to the projected patches.
    encoded_patches = position_embedding(
        projected_patches
    )  # (B, number_patches, projection_dim)
    encoded_patches = layers.Dropout(0.1)(encoded_patches)

    # Iterate over the number of layers and stack up blocks of
    # Transformer.
    for i in range(NUM_LAYERS):
        # Add a Transformer block.
        print("transformer block", i)
        encoded_patches = transformer(encoded_patches)

        # Add TokenLearner layer in the middle of the
        # architecture. The paper suggests that anywhere
        # between 1/2 or 3/4 will work well.
        if use_token_learner and i == NUM_LAYERS // 2:
            _, hh, c = encoded_patches.shape
            print( hh, c)

            print("reshaping to patches")

            encoded_patches = layers.Reshape((18, 10, c))(encoded_patches)
            print("encoded_patches.shape", encoded_patches.shape)
            #encoded_patches = layers.Reshape((h, h, c))(
            #    encoded_patches
            #)  # (B, h, h, projection_dim)
            encoded_patches = token_learner(
                encoded_patches, token_learner_units
            )  # (B, num_tokens, c)

    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(encoded_patches)
    print("representation.shape after layerNorm", representation.shape)
    representation = layers.GlobalAvgPool1D()(representation)
    print("representation.shape, after GAP", representation.shape)
    # Classify outputs.
    outputs = layers.Dense(1, activation="relu")(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

#
# #As shown in the TokenLearner paper, it is almost always advantageous to include the TokenLearner module in the middle of the network.
# #Training utility
# def run_experiment(model):
#     # Initialize the AdamW optimizer.
#     optimizer = tfa.optimizers.AdamW(
#         learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
#     )
#
#     # Compile the model with the optimizer, loss function
#     # and the metrics.
#     model.compile(
#         optimizer=optimizer,
#         loss="sparse_categorical_crossentropy",
#         metrics=[
#             keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
#             keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
#         ],
#     )
#
#     # Define callbacks
#     checkpoint_filepath = "/tmp/checkpoint"
#     checkpoint_callback = keras.callbacks.ModelCheckpoint(
#         checkpoint_filepath,
#         monitor="val_accuracy",
#         save_best_only=True,
#         save_weights_only=True,
#     )
#
#     # Train the model.
#     _ = model.fit(lege
#         train_ds,
#         epochs=EPOCHS,
#         validation_data=val_ds,
#         callbacks=[checkpoint_callback],
#     )
#
#     model.load_weights(checkpoint_filepath)
#     _, accuracy, top_5_accuracy = model.evaluate(test_ds)
#     print(f"Test accuracy: {round(accuracy * 100, 2)}%")
#     print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
# #Train and evaluate a ViT with TokenLearner
# vit_token_learner = create_vit_classifier()
# run_experiment(vit_token_learner)
