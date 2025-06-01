import os
import re
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import resample
import yaml
import requests

# Load config.yaml
def load_config(path='config.yaml'):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def data_loading(root, category):
    folder_path = root + f"/{category}"

    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv") and f!=f"{category}.csv"]

    # List to store each DataFrame
    dfs = []

    # Read and collect DataFrames
    for file in csv_files:
        full_path = os.path.join(folder_path, file)
        # print(f"Reading: {full_path}")
        df = pd.read_csv(full_path)
        # df['item_id'] = 
        # print(df.head(5))
        dfs.append(df)
    
    # Combine all into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    # Show result
    # print(combined_df.head())
    print(f"\nTotal combined rows of {category}: {len(combined_df)}")

    return combined_df

def balance_df(df):
    # Separate by class
    df_0 = df[df['label'] == 'negative']
    df_1 = df[df['label'] == 'neutral']
    df_2 = df[df['label'] == 'positive']

    # Find the smallest class size
    min_class_size = min(len(df_0), len(df_1), len(df_2))

    # Generate random multipliers for each class
    mult_0 = np.random.uniform(1.0, 1.5)
    mult_1 = np.random.uniform(1.0, 1.5)
    mult_2 = np.random.uniform(1.0, 1.5)

    # Compute target sample sizes
    n_0 = min(len(df_0), int(min_class_size * mult_0))
    n_1 = min(len(df_1), int(min_class_size * mult_1))
    n_2 = min(len(df_2), int(min_class_size * mult_2))

    # Downsample each class
    df_0_down = resample(df_0, replace=False, n_samples=n_0, random_state=42)
    df_1_down = resample(df_1, replace=False, n_samples=n_1, random_state=43)
    df_2_down = resample(df_2, replace=False, n_samples=n_2, random_state=44)

    # Combine and shuffle
    df_balanced = pd.concat([df_0_down, df_1_down, df_2_down])
    df_balanced = df_balanced.sample(frac=1, random_state=45).reset_index(drop=True)

    # View new distribution
    print("Random multipliers:", f"Class 0: {mult_0:.2f}, Class 1: {mult_1:.2f}, Class 2: {mult_2:.2f}")
    print(df_balanced['label'].value_counts())

    return df_balanced

def data_prep(df, vocab_size, max_length):
    review = df['text_akhir']

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(review)

    sequences = tokenizer.texts_to_sequences(review)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    return padded_sequences, tokenizer

def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return ""

    text_clean = cleaningText(text)
    text_casefolded = casefoldingText(text_clean)
    text_stemmed = stemmingText(text_casefolded)
    text_slang_fixed = fix_slangwords(text_stemmed)
    text_tokenized = tokenizingText(text_slang_fixed)
    text_filtered = filteringText(text_tokenized)
    text_final = toSentence(text_filtered)

    return text_final

#####################################################################################
def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # menghapus mention
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # menghapus hashtag
    text = re.sub(r'RT[\s]', '', text) # menghapus RT
    text = re.sub(r"http\S+", '', text) # menghapus link
    text = re.sub(r'[0-9]+', '', text) # menghapus angka
    text = re.sub(r'[^\w\s]', '', text) # menghapus karakter selain huruf dan angka
 
    text = text.replace('\n', ' ') # mengganti baris baru dengan spasi
    text = text.translate(str.maketrans('', '', string.punctuation)) # menghapus semua tanda baca
    text = text.strip(' ') # menghapus karakter spasi dari kiri dan kanan teks

    # remove emojis
    emoji_pattern = re.compile(
        "["                               
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002700-\U000027BF"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    text = emoji_pattern.sub(r'', text)
    
    return text
 
def casefoldingText(text): # Mengubah semua karakter dalam teks menjadi huruf kecil
    text = text.lower()
    return text
 
# def tokenizingText(text): # Memecah atau membagi string, teks menjadi daftar token
#     text = word_tokenize(text)
#     return text
 
def tokenizingText(text):
    if pd.isna(text):
        return []
    return text.split()

def filteringText(text): # Menghapus stopwords dalam teks
    listStopwords = set(stopwords.words('indonesian'))
    listStopwords1 = set(stopwords.words('english'))
    listStopwords.update(listStopwords)
    # listStopwords.update(listStopwords1)
    listStopwords.update(['iya','yaa','nya','na','sih','ku',"di","ya","loh","kah","woi","woii","woy", "nih", "trus", "tuh",\
                          "yah", "ajah", "lagi", "lah", "aj", "aja", "jg", "juga", "jga", "jugaa", "yng", 'apa', "cuman", "deh",\
                            "min", "gak", "cuma",\
                            "si", "an", "dikit", "langsung"])
    filtered = []
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered
    return text
 
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemmingText(text):
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_text = ' '.join(stemmed_words)
    return stemmed_text
 
def toSentence(list_words): # Mengubah daftar kata menjadi kalimat
    sentence = ' '.join(word for word in list_words)
    return sentence

# Download slang dictionary from a public dataset (if available)
url = "https://raw.githubusercontent.com/louisowen6/NLP_bahasa_resources/refs/heads/master/combined_slang_words.txt"
slang_dict = requests.get(url).json()

slang_dict['aku'] = 'saya'
slang_dict['ak'] = 'saya'
slang_dict['gua'] = 'saya'
slang_dict['gw'] = 'saya'

slang_dict['jlk'] = 'jelek'
slang_dict['jlek'] = 'jelek'
slang_dict['burik'] = 'jelek'
slang_dict['buriq'] = 'jelek'
slang_dict['ampas'] = 'jelek'
slang_dict['amps'] = 'jelek'
slang_dict['buruk'] = 'jelek'
slang_dict['kentang'] = 'jelek'
slang_dict['bobrok'] = 'jelek'

slang_dict['bgs'] = 'bagus'
slang_dict['wokeh'] = 'bagus'
slang_dict['bgus'] = 'bagus'
slang_dict['baguss'] = 'bagus'

slang_dict['trnyata'] = 'ternyata'

slang_dict['amann'] = 'aman'

slang_dict['syukaa'] = 'suka'

slang_dict['bgt'] = 'banget'
slang_dict['bgtt'] = 'banget'

slang_dict['kren'] = 'keren'

slang_dict['udh'] = 'udah'

slang_dict['kasi'] = 'kasih'
slang_dict['ksi'] = 'kasih'
slang_dict['ksih'] = 'kasih'

slang_dict['gk'] = 'gak'
slang_dict['ga'] = 'gak'
slang_dict['gaa'] = 'gak'
slang_dict['kagak'] = 'gak'
slang_dict['kgk'] = 'gak'
slang_dict['g'] = 'gak'
slang_dict['engga'] = 'gak'
slang_dict['tdk'] = 'gak'
slang_dict['nggk'] = 'gak'
slang_dict['no'] = 'gak'

slang_dict['jls'] = 'jelas'
slang_dict['jlas'] = 'jelas'
slang_dict['danta'] = 'jelas'

slang_dict['mntp'] = 'mantap'
slang_dict['mantul'] = 'mantap'
slang_dict['mntap'] = 'mantap'

slang_dict['lg'] = 'lagi'
slang_dict['lgi'] = 'lagi'

slang_dict['uk'] = 'ukuran'

slang_dict['ksel'] = 'kesal'
slang_dict['kesel'] = 'kesal'
slang_dict['sebel'] = 'kesal'
slang_dict['sebal'] = 'kesal'

slang_dict['bacod'] = 'bacot'
slang_dict['bct'] = 'bacot'
slang_dict['bcd'] = 'bacot'

slang_dict['goblog'] = 'goblok'
slang_dict['gblg'] = 'goblok'
slang_dict['gblk'] = 'goblok'
slang_dict['bego'] = 'goblok'
slang_dict['bgo'] = 'goblok'
slang_dict['tolol'] = 'goblok'
slang_dict['tlol'] = 'goblok'
slang_dict['idiot'] = 'goblok'

slang_dict['trun'] = 'turun'

slang_dict['brg'] = 'barang'
slang_dict['brang'] = 'barang'
slang_dict['barng'] = 'barang'

slang_dict['cm'] = 'cuma'
slang_dict['cma'] = 'cuma'
slang_dict['cman'] = 'cuma'
slang_dict['cmn'] = 'cuma'

slang_dict['yt'] = 'youtube'

slang_dict['wrnaa'] = 'warna'

slang_dict['ajg'] = 'anjing'
slang_dict['anj'] = 'anjing'
slang_dict['anjg'] = 'anjing'
slang_dict['anjir'] = 'anjing'
slang_dict['anjr'] = 'anjing'

slang_dict['leg'] = 'lambat'
slang_dict['ngeleg'] = 'lambat'
slang_dict['lemod'] = 'lambat'
slang_dict['lemot'] = 'lambat'

slang_dict['happy'] = 'senang'

slang_dict['satset'] = 'cepat'
slang_dict['cpt'] = 'cepat'

slang_dict['pass'] = 'pas'

slang_dict['sbg'] = 'sebagai'

slang_dict['wr'] = 'win rate'
slang_dict['winrate'] = 'win rate'
slang_dict['ws'] = 'win streak'
slang_dict['winstreak'] = 'win streak'

slang_dict['ori'] = 'asli'
slang_dict['original'] = 'asli'

slang_dict['kw'] = 'palsu'
slang_dict['fake'] = 'palsu'

slang_dict['ok'] = 'oke'
slang_dict['okey'] = 'oke'
slang_dict['okay'] = 'oke'

slang_dict['hps'] = 'hapus'
slang_dict['hpus'] = 'hapus'
slang_dict['uninstal'] = 'hapus'
slang_dict['uninstall'] = 'hapus'

slang_dict['dikirim'] = 'pengiriman'

# Bi-gram
# Common
slang_dict['cepat selesai'] = 'cepat'

slang_dict['gak palsu'] = 'asli'
slang_dict['gak asli'] = 'palsu'
slang_dict['gak jelas'] = 'aneh'
slang_dict['gaje'] = 'aneh'

slang_dict['suka banget'] = 'cinta'

slang_dict['tebel'] = 'tebal'

slang_dict['gak suka'] = 'jelek'
slang_dict['gak enak'] = 'jelek'
slang_dict['gak bagus'] = 'jelek'

slang_dict['murah banget'] = 'murah_banget'

slang_dict['mahal banget'] = 'mahal_banget'

slang_dict['cepet banget'] = 'cepet_banget'

slang_dict['lama banget'] = 'lambat_banget'
slang_dict['lambat banget'] = 'lambat_banget'

slang_dict['bagus banget'] = 'bagus_banget'

slang_dict['jelek banget'] = 'jelek_banget'
slang_dict['sangat jelek'] = 'jelek_banget'

slang_dict['pelayanan buruk'] = 'buruk'

slang_dict['sangat puas'] = 'sangat_puas'

slang_dict['gak puas'] = 'kecewa'

# Purchase & Delivery
slang_dict['barang datang'] = "datang"
slang_dict['barang telat'] = "lambat"
slang_dict['barang cepat'] = "cepat"
slang_dict['barang rusak'] = "rusak"
slang_dict['barang oke'] = "bagus"
slang_dict['tebal banget'] = "bagus"
slang_dict['barang bagus'] = "bagus"
slang_dict['barang jelek'] = "jelek"
slang_dict['pengiriman cepat'] = "cepat"
slang_dict['pengiriman lambat'] = "lambat"
slang_dict['pengiriman aman'] = "aman"
slang_dict['pengiriman oke'] = "bagus"

def fix_slangwords(text):
    words = text.lower().split()
    
    # Step 1: Fix unigrams
    fixed_unigrams = [slang_dict.get(word, word) for word in words]
    
    # Step 2: Check for fixed bigrams
    i = 0
    final_words = []
    while i < len(fixed_unigrams):
        if i + 1 < len(fixed_unigrams):
            bigram = f"{fixed_unigrams[i]} {fixed_unigrams[i+1]}"
            if bigram in slang_dict:
                final_words.append(slang_dict[bigram])
                i += 2
                continue
        
        final_words.append(fixed_unigrams[i])
        i += 1

    return ' '.join(final_words)