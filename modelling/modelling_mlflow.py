import  mlflow
import pandas as pd
import os
import tensorflow as tf
import re
import string
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import requests
import csv
from io import StringIO
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from rapidfuzz import process
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from mlflow.models.signature import infer_signature
from sklearn.metrics import precision_recall_fscore_support
from keras.saving import register_keras_serializable
from custom_layers import TransformerBlock, TokenAndPositionEmbedding
from utils import data_loading, data_prep, balance_df, load_config

def build_model(embed_dim, num_heads, ff_dim, maxlen, vocab_size,\
    num_transformer_blocks, dropout1_rate, dropout2_rate, dropout3_rate, \
    dense_units, num_classes):

    inputs = Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    x = Dropout(dropout1_rate)(x)

    # Multiple transformer blocks
    for _ in range(num_transformer_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dropout(dropout2_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Dense(dense_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(dropout3_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == '__main__':
    config = load_config()

    # Access hyperparameters
    dataset_ver = config['dataset']['version']
    maxlen = config['tokenizer']['maxlen']
    vocab_size = config['tokenizer']['vocab_size']

    num_transformer_blocks = config['model']['num_transformer_blocks']
    embed_dim = config['model']['embed_dim']
    num_heads = config['model']['num_heads']
    ff_dim = config['model']['ff_dim']
    dense_units = config['model']['dense_units']
    num_classes = config['model']['num_classes']

    dropout1_rate = config['dropout']['dropout1_rate']
    dropout2_rate = config['dropout']['dropout2_rate']
    dropout3_rate = config['dropout']['dropout3_rate']

    root_path = ""
    if dataset_ver == 2:
        root_path = "../data_preprocessing/dataset_preprocessed_v2"
    else:
        root_path = "../data_preprocessing/dataset_preprocessed"

    # List all CSV files in the folder
    category_names = [f for f in os.listdir(root_path)]

    print(category_names)

    # Combine DF
    all_dfs = {
        name: data_loading(root_path, category)
        for name, category in zip(category_names, category_names)
    }

    for name, df in all_dfs.items():
        print(f"Category: {name}")
        df = balance_df(df.dropna())
        all_dfs[name] = df

    combined_df = pd.concat(all_dfs.values(), ignore_index=True)

    # Data Preparation
    text_pad, tokenizer = data_prep(combined_df, vocab_size, maxlen)

    X = np.array(text_pad)

    # Convert labels to integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(combined_df['label'])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # ML Flow Tracking
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")
    mlflow.set_experiment('Dataset_Stemmed')
    mlflow.tensorflow.autolog()

    # Model building
    model = build_model(embed_dim, num_heads, ff_dim, maxlen, vocab_size, num_transformer_blocks,\
                        dropout1_rate, dropout2_rate, dropout3_rate, dense_units, num_classes)

    mlflow.log_param("embed_dim", embed_dim)
    mlflow.log_param("num_heads", num_heads)
    mlflow.log_param("ff_dim", ff_dim)
    mlflow.log_param("maxlen", maxlen)
    mlflow.log_param("vocab_size", vocab_size)
    mlflow.log_param("num_transformer_blocks", num_transformer_blocks)
    mlflow.log_param("dropout1_rate", dropout1_rate)
    mlflow.log_param("dropout2_rate", dropout2_rate)
    mlflow.log_param("dropout3_rate", dropout3_rate)
    mlflow.log_param("dense_units", dense_units)
    mlflow.log_param("num_classes", num_classes)

    ### COMPILE ###
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

    model.summary()

    # Define EarlyStopping callback
    early_stop = EarlyStopping(
        monitor='val_accuracy',  # Monitor validation accuracy
        patience=5,              # Stop after 5 epochs with no improvement
        restore_best_weights=True  # Restore the best model weights
    )

    with mlflow.start_run(nested=True) as run:
        # Train the model with EarlyStopping
        model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,
            validation_data=(X_val, y_val),
            callbacks=[early_stop]
        )

        # mlflow.sklearn.log_model(model, "model", input_example=X_train[:5])

        signature = infer_signature(X_train[:5], model.predict(X_train[:5]))

        # custom_objects = {
        #     "TransformerBlock": TransformerBlock,
        #     "TokenAndPositionEmbedding": TokenAndPositionEmbedding
        # }

        mlflow.keras.log_model(model, "model", signature=signature)

        print(f"Run ID: {run.info.run_id}")

        ### Metrics Logging ###
        # Predict on validation or test set
        y_pred = model.predict(X_val).argmax(axis=1)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_val, y_pred, average=None, labels=[0, 1, 2])

        # Log per-class metrics
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            mlflow.log_metric(f"precision_class_{i}", p)
            mlflow.log_metric(f"recall_class_{i}", r)
            mlflow.log_metric(f"f1_score_class_{i}", f)