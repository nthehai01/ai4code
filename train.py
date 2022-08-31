import tensorflow as tf

from model.model import Model
from constant import *
from data import Dataset


###############################################################################
# DATA PREPARATION
###############################################################################

d = Dataset(
    DATA_DIR, 
    MODEL_PATH, 
    MAX_LEN, 
    MAX_CELL, 
    NUM_TRAIN, 
    BUFFER_SIZE, 
    BATCH_SIZE, 
    CELL_PAD
)
train = d.build_dataset()


###############################################################################
# MODEL
###############################################################################

# Input definitions
input_ids = tf.keras.layers.Input(shape=(MAX_CELL, MAX_LEN), name='input_ids', dtype='int32')
attention_mask = tf.keras.layers.Input(shape=(MAX_CELL, MAX_LEN), name='attention_mask', dtype='int32')
cell_features = tf.keras.layers.Input(shape=(MAX_CELL, 2), name='cell_features', dtype='float32')
cell_mask = tf.keras.layers.Input(shape=(MAX_CELL, 1), name='cell_mask', dtype='int32')

# Model initialization
stacked_transformer = Model(
    model_path=MODEL_PATH, 
    d_model=D_MODEL, 
    d_ff_pool=512, 
    n_heads=8, 
    dropout_trans=0.2, 
    eps=1e-6, 
    d_ff_trans=2048, 
    ff_activation="relu", 
    n_layers=6,
    d_ff_pointwise=512, 
    dropout_pointwise=0.2
)

# Define the model
model = tf.keras.Model(
    inputs=[input_ids, attention_mask, cell_features], 
    outputs=stacked_transformer.call(input_ids, attention_mask, cell_features, True)
)

# Model summary
model.summary()


###############################################################################
# TRAINING
###############################################################################

optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
loss = tf.keras.losses.MeanAbsoluteError()

model.compile(optimizer=optimizer, loss=loss)

history = model.fit(train, epochs=1)
