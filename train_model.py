import json
import pickle
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

from chatbot_engine import preprocess_text, load_json_dataset

SINONIM_INDONESIA = {
    'apa': ['apakah', 'apa sih', 'apa ya', 'apaan', 'apa itu'],
    'itu': ['tersebut', 'ini', 'tuh', 'tsb'],
    'cara': ['bagaimana', 'gimana', 'caranya', 'langkah', 'tutorial', 'step'],
    'bagaimana': ['gimana', 'cara', 'caranya', 'bagaimanakah', 'gmn'],
    'gimana': ['bagaimana', 'cara', 'caranya', 'gmn', 'gmna'],
    'untuk': ['buat', 'utk', 'guna', 'supaya', 'tuk'],
    'buat': ['untuk', 'utk', 'bikin', 'membuat'],
    'pakai': ['menggunakan', 'gunakan', 'pake', 'aplikasikan', 'pkai'],
    'menggunakan': ['pakai', 'pake', 'gunakan', 'memakai'],
    'harga': ['berapa', 'biaya', 'cost', 'hrg', 'price'],
    'berapa': ['harga', 'brp', 'brapa', 'berapaan'],
    'fungsi': ['kegunaan', 'manfaat', 'guna', 'khasiat'],
    'manfaat': ['fungsi', 'kegunaan', 'guna', 'khasiat', 'benefit'],
    'kegunaan': ['fungsi', 'manfaat', 'guna'],
    'jenis': ['macam', 'tipe', 'ragam', 'variasi'],
    'macam': ['jenis', 'tipe', 'ragam'],
    'kulit': ['wajah', 'muka', 'skin'],
    'wajah': ['muka', 'kulit', 'face'],
    'muka': ['wajah', 'kulit'],
    'berminyak': ['oily', 'minyakan', 'licin', 'kilap'],
    'kering': ['dry', 'kasar', 'kusam'],
    'jerawat': ['acne', 'bruntusan', 'breakout', 'pimple'],
    'produk': ['barang', 'item', 'product'],
    'rekomendasi': ['saran', 'rekomen', 'suggest', 'recommend', 'rekomendasikan'],
    'bagus': ['baik', 'oke', 'cocok', 'recommended', 'mantap'],
    'terbaik': ['best', 'paling bagus', 'nomor satu', 'top'],
    'beli': ['buat', 'order', 'pesan', 'checkout'],
    'halo': ['hai', 'hi', 'hello', 'hei', 'hey', 'p', 'permisi'],
    'hai': ['halo', 'hi', 'hello', 'hei', 'p'],
    'terima kasih': ['makasih', 'thanks', 'thank you', 'thx', 'trims'],
    'makasih': ['terima kasih', 'thanks', 'thx', 'trims'],
    'tolong': ['mohon', 'bantu', 'help', 'minta', 'tlg'],
    'tidak': ['nggak', 'gak', 'ga', 'enggak', 'tdk', 'kagak'],
    'nggak': ['tidak', 'gak', 'ga', 'enggak', 'kagak'],
    'ada': ['punya', 'tersedia', 'available'],
    'promo': ['diskon', 'potongan', 'sale', 'discount'],
    'diskon': ['promo', 'potongan', 'sale', 'discount'],
    'murah': ['terjangkau', 'affordable', 'hemat', 'ekonomis'],
    'mahal': ['premium', 'high-end', 'mewah'],
    'cocok': ['pas', 'sesuai', 'tepat', 'match'],
    'masalah': ['problem', 'issue', 'kendala', 'gangguan'],
    'solusi': ['jawaban', 'cara', 'jalan keluar', 'solution'],
    'butuh': ['perlu', 'memerlukan', 'need', 'mau'],
    'cari': ['mencari', 'nyari', 'looking', 'search'],
    'tanya': ['bertanya', 'nanya', 'ask', 'mau tau'],
    'kasih': ['beri', 'berikan', 'give', 'tolong'],
    'info': ['informasi', 'keterangan', 'detail', 'penjelasan'],
    'cepat': ['kilat', 'express', 'segera', 'buruan'],
    'lama': ['lambat', 'slow', 'butuh waktu'],
    'baru': ['new', 'terbaru', 'latest', 'fresh'],
    'lama': ['old', 'lama', 'dulu'],
}

VARIASI_INFORMAL = [
    ('apa', ['apaan', 'apasih', 'apa sih', 'apa ya', 'apakah', 'ap']),
    ('bagaimana', ['gimana', 'gmn', 'gmna', 'gimanasih', 'caranya']),
    ('kenapa', ['knp', 'knpa', 'napa', 'ngapa', 'why']),
    ('dimana', ['dmn', 'dmna', 'mana', 'd mana']),
    ('kapan', ['kpn', 'kpan', 'when']),
    ('siapa', ['sapa', 'who', 'siapasih']),
    ('berapa', ['brp', 'brpa', 'brapa', 'berapaan']),
    ('bisa', ['bs', 'bsa', 'dapat', 'bole', 'boleh']),
    ('harus', ['hrs', 'hrus', 'kudu', 'wajib', 'mesti']),
    ('ingin', ['mau', 'pengen', 'pgn', 'pingin', 'mw']),
    ('sudah', ['udah', 'udh', 'sdh', 'uda', 'dah']),
    ('belum', ['blm', 'blom', 'blum', 'belon']),
    ('tidak', ['gak', 'ga', 'nggak', 'ngga', 'tdk', 'kagak', 'kaga']),
    ('dengan', ['dgn', 'dg', 'sama', 'pake']),
    ('yang', ['yg', 'yng']),
    ('juga', ['jg', 'jga']),
    ('saja', ['aja', 'aj', 'doang', 'doank']),
    ('lagi', ['lg', 'lgi']),
]

def generate_typo_variations(text):
    variations = [text]
    words = text.split()
    for i, word in enumerate(words):
        if len(word) > 3:
            new_words = words.copy()
            new_words[i] = word + word[-1]
            variations.append(' '.join(new_words))
            new_words = words.copy()
            new_words[i] = word + word[-1] + word[-1]
            variations.append(' '.join(new_words))
    return variations

def generate_informal_variations(text):
    variations = [text]
    text_lower = text.lower()
    for formal, informals in VARIASI_INFORMAL:
        if formal in text_lower:
            for informal in informals[:2]:
                new_text = text_lower.replace(formal, informal)
                variations.append(new_text)
    return variations

def augment_text(text):
    augmented = set()
    augmented.add(text)
    words = text.lower().split()
    for i, word in enumerate(words):
        if word in SINONIM_INDONESIA:
            for synonym in SINONIM_INDONESIA[word][:3]:
                new_words = words.copy()
                new_words[i] = synonym
                augmented.add(' '.join(new_words))
    if len(words) <= 5:
        question_words = ['apa', 'apakah', 'bagaimana', 'gimana', 'berapa', 'kapan', 'dimana', 'kenapa', 'siapa']
        for qw in question_words:
            if text.lower().startswith(qw + ' '):
                remaining = text[len(qw)+1:].strip()
                augmented.add(remaining)
                augmented.add(remaining + ' ' + qw)
                augmented.add(qw + ' ' + remaining)
    typo_vars = generate_typo_variations(text)
    for var in typo_vars:
        augmented.add(var)
    informal_vars = generate_informal_variations(text)
    for var in informal_vars:
        augmented.add(var)
    if len(words) >= 2:
        augmented.add(' '.join(words[::-1]))
    if len(words) > 2:
        for i in range(len(words)):
            shorter = words[:i] + words[i+1:]
            if len(shorter) >= 1:
                augmented.add(' '.join(shorter))
    return list(augmented)

def load_dataset(filepath, augment=True):
    data = load_json_dataset(filepath)
    patterns = []
    labels = []
    for intent in data['intents']:
        tag = intent['tag']
        for pattern in intent['patterns']:
            if augment:
                augmented_patterns = augment_text(pattern)
                for aug_pattern in augmented_patterns:
                    processed = preprocess_text(aug_pattern)
                    if processed.strip() and len(processed) >= 2:
                        patterns.append(processed)
                        labels.append(tag)
            else:
                processed = preprocess_text(pattern)
                if processed.strip():
                    patterns.append(processed)
                    labels.append(tag)
    return patterns, labels

def train_model(X_train, X_test, y_train, y_test):
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 4),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        analyzer='char_wb',
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    model = LogisticRegression(
        max_iter=2000,
        random_state=42,
        C=10.0,
        class_weight='balanced',
        solver='lbfgs',
    )
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi Model: {accuracy:.4f}")
    return vectorizer, model

def save_model(vectorizer, model, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(f'{output_dir}/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"Model disimpan di folder {output_dir}")

def main():
    print("Beauty Paw Chatbot - Melatih Model")
    dataset_path = 'datasets.json'
    patterns, labels = load_dataset(dataset_path, augment=True)
    X_train, X_test, y_train, y_test = train_test_split(
        patterns, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    vectorizer, model = train_model(X_train, X_test, y_train, y_test)
    save_model(vectorizer, model)
    print("Pelatihan selesai")

if __name__ == "__main__":
    main()
