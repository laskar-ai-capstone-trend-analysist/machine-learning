import os
import pandas as pd
from tensorflow.keras.models import load_model
from custom_layers import TransformerBlock, TokenAndPositionEmbedding
from utils import data_loading, data_prep, load_config, preprocess_text
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

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

all_dfs = {
    name: data_loading(root_path, category)
    for name, category in zip(category_names, category_names)
}

for name, df in all_dfs.items():
    print(f"Category: {name}")
    df = df.dropna()
    all_dfs[name] = df

combined_df = pd.concat(all_dfs.values(), ignore_index=True)

text_pad, tokenizer = data_prep(combined_df, vocab_size, maxlen)

# Load the model
loaded_model = load_model("sentiment_model.keras")

loaded_model.summary()

### TESTING 1
text_testing1 = 'brng jelek, pengiriman cepat'
text_testing1 = preprocess_text(text_testing1)

# Convert to sequence using the same tokenizer
test_seq = tokenizer.texts_to_sequences([text_testing1])

# Pad it to the same max length
test_pad = pad_sequences(test_seq, maxlen, padding='post')

prediction = loaded_model.predict(test_pad)

# Get class with highest probability
predicted_class = np.argmax(prediction, axis=1)[0]

print(f"Text: {text_testing1}")
print(f"Predicted label index: {predicted_class}")

### TESTING 2
text_testing2 = 'barang bagus sekali bagus banget bagus banget, ga fake'
text_testing2 = preprocess_text(text_testing2)

# Convert to sequence using the same tokenizer
test_seq = tokenizer.texts_to_sequences([text_testing2])

# Pad it to the same max length
test_pad = pad_sequences(test_seq, maxlen, padding='post')

prediction = loaded_model.predict(test_pad)

# Get class with highest probability
predicted_class = np.argmax(prediction, axis=1)[0]

print(f"Text: {text_testing2}")
print(f"Predicted label index: {predicted_class}")