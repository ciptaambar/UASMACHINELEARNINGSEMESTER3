import re
import json
import pickle
import random
from difflib import SequenceMatcher

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()
    USE_STEMMER = True
except ImportError:
    USE_STEMMER = False
    stemmer = None

SLANG_DICTIONARY = {
    'gw': 'saya', 'gue': 'saya', 'gua': 'saya', 'aku': 'saya', 'ak': 'saya', 'w': 'saya',
    'lu': 'kamu', 'lo': 'kamu', 'elu': 'kamu', 'elo': 'kamu', 'u': 'kamu', 'km': 'kamu', 'kmu': 'kamu',
    'gmn': 'gimana', 'gmna': 'gimana', 'gimana': 'bagaimana',
    'gmw': 'tidak mau',
    'gak': 'tidak', 'ga': 'tidak', 'g': 'tidak', 'gk': 'tidak', 'nggak': 'tidak', 'ngga': 'tidak', 'enggak': 'tidak', 'engga': 'tidak', 'tdk': 'tidak',
    'gpp': 'tidak apa-apa', 'gapapa': 'tidak apa-apa',
    'yg': 'yang', 'yng': 'yang',
    'utk': 'untuk', 'tuk': 'untuk',
    'bwt': 'buat', 'bt': 'buat',
    'dgn': 'dengan', 'dg': 'dengan',
    'sm': 'sama', 'sma': 'sama',
    'org': 'orang', 'ornag': 'orang', 'orng': 'orang',
    'jg': 'juga', 'jga': 'juga',
    'jd': 'jadi', 'jdi': 'jadi',
    'udh': 'sudah', 'udah': 'sudah', 'sdh': 'sudah', 'uda': 'sudah', 'dah': 'sudah',
    'blm': 'belum', 'blom': 'belum', 'blum': 'belum',
    'tp': 'tapi', 'tpi': 'tapi',
    'ttg': 'tentang',
    'bgt': 'banget', 'bngt': 'banget', 'bngtt': 'banget', 'banget': 'sangat',
    'bener': 'benar', 'bnr': 'benar',
    'emg': 'memang', 'emang': 'memang', 'emng': 'memang',
    'hrs': 'harus', 'hrus': 'harus',
    'bs': 'bisa', 'bsa': 'bisa',
    'aja': 'saja', 'aj': 'saja', 'doang': 'saja', 'doank': 'saja',
    'lg': 'lagi', 'lgi': 'lagi',
    'skrg': 'sekarang', 'skr': 'sekarang', 'skg': 'sekarang', 'skrang': 'sekarang',
    'bsk': 'besok', 'besok': 'besok',
    'kmrn': 'kemarin', 'kmrin': 'kemarin',
    'knp': 'kenapa', 'knpa': 'kenapa', 'napa': 'kenapa', 'ngapa': 'kenapa',
    'ngapain': 'sedang apa',
    'apaan': 'apa', 'apa sih': 'apa', 'apasih': 'apa', 'ap': 'apa',
    'kyk': 'kayak', 'kek': 'kayak', 'kayak': 'seperti',
    'klo': 'kalau', 'kalo': 'kalau', 'kl': 'kalau', 'klau': 'kalau',
    'mksd': 'maksud', 'mksud': 'maksud', 'mksdnya': 'maksudnya',
    'dr': 'dari', 'dri': 'dari',
    'pd': 'pada', 'pda': 'pada',
    'spy': 'supaya', 'biar': 'supaya',
    'bnyk': 'banyak', 'byk': 'banyak',
    'sdikit': 'sedikit', 'sdkit': 'sedikit', 'dkit': 'sedikit', 'dikit': 'sedikit',
    'bgmn': 'bagaimana', 'bgmna': 'bagaimana',
    'dmn': 'dimana', 'dmna': 'dimana',
    'mna': 'mana',
    'kpn': 'kapan',
    'trs': 'terus', 'trus': 'terus', 'trz': 'terus',
    'abis': 'habis', 'abs': 'habis', 'hbs': 'habis',
    'pake': 'pakai', 'pk': 'pakai', 'pke': 'pakai', 'pkai': 'pakai',
    'sy': 'saya',
    'thx': 'terima kasih', 'thanks': 'terima kasih', 'thank': 'terima kasih', 'tq': 'terima kasih', 'makasih': 'terima kasih', 'makasi': 'terima kasih', 'mksh': 'terima kasih', 'mkasih': 'terima kasih', 'mksih': 'terima kasih', 'trims': 'terima kasih', 'trimakasih': 'terima kasih', 'trmksh': 'terima kasih',
    'ok': 'oke', 'okay': 'oke', 'okey': 'oke', 'oks': 'oke', 'okee': 'oke', 'okeee': 'oke', 'siap': 'oke', 'sip': 'oke', 'sippp': 'oke',
    'mantap': 'bagus', 'mantep': 'bagus', 'mantab': 'bagus', 'keren': 'bagus', 'cakep': 'bagus',
    'btw': 'ngomong-ngomong',
    'fyi': 'untuk informasi',
    'info': 'informasi',
    'krn': 'karena', 'krna': 'karena', 'soalnya': 'karena', 'coz': 'karena', 'cuz': 'karena', 'cos': 'karena',
    'pdhl': 'padahal', 'pdhal': 'padahal',
    'bbrp': 'beberapa',
    'brp': 'berapa', 'brpa': 'berapa', 'brapa': 'berapa',
    'hrg': 'harga', 'hrga': 'harga',
    'dpt': 'dapat', 'dpat': 'dapat', 'dapet': 'dapat',
    'td': 'tadi', 'tdi': 'tadi',
    'jgn': 'jangan', 'jngn': 'jangan', 'jng': 'jangan',
    'smpe': 'sampai', 'smp': 'sampai', 'sampe': 'sampai', 'ampe': 'sampai',
    'msh': 'masih', 'msih': 'masih', 'masi': 'masih',
    'cb': 'coba', 'cba': 'coba',
    'tlg': 'tolong', 'tlng': 'tolong', 'pls': 'tolong', 'please': 'tolong', 'pliss': 'tolong', 'plis': 'tolong',
    'mo': 'mau', 'mw': 'mau',
    'pengen': 'ingin', 'pgn': 'ingin', 'pngen': 'ingin', 'pengin': 'ingin', 'pingin': 'ingin', 'pgnn': 'ingin',
    'dl': 'dulu', 'dlu': 'dulu',
    'ntr': 'nanti', 'ntar': 'nanti', 'nnti': 'nanti', 'tar': 'nanti',
    'trnyata': 'ternyata',
    'wkt': 'waktu', 'wktu': 'waktu',
    'bgtu': 'begitu', 'gtu': 'begitu', 'gitu': 'begitu',
    'bgni': 'begini', 'gni': 'begini', 'gini': 'begini',
    'sbg': 'sebagai', 'sbgai': 'sebagai',
    'spt': 'seperti', 'sprti': 'seperti',
    'kyg': 'yang',
    'ygmana': 'yang mana',
    'kemana': 'ke mana', 'kmana': 'ke mana', 'kmn': 'ke mana',
    'mnrt': 'menurut', 'menurut': 'menurut',
    'sbnernya': 'sebenarnya', 'sbnrnya': 'sebenarnya', 'sbnarnya': 'sebenarnya', 'sbnrny': 'sebenarnya',
    'bnran': 'beneran', 'bneran': 'beneran',
    'serius': 'serius', 'srius': 'serius', 'srs': 'serius',
    'wkwk': '', 'wkwkwk': '', 'haha': '', 'hahaha': '', 'hihi': '', 'hehe': '', 'xixi': '', 'kwkw': '', 'lol': '', 'lmao': '',
    'hi': 'hai', 'hii': 'hai', 'hiii': 'hai',
    'hello': 'halo', 'helo': 'halo',
    'hey': 'hai', 'hei': 'hai',
    'p': 'halo', 'pp': 'halo',
    'hai': 'halo', 'haii': 'halo', 'haiii': 'halo',
    'assalamualaikum': 'halo', 'assalamu': 'halo', 'asslm': 'halo', 'aslm': 'halo', 'slm': 'halo',
    'pagi': 'selamat pagi', 'pagii': 'selamat pagi',
    'siang': 'selamat siang',
    'sore': 'selamat sore',
    'malam': 'selamat malam', 'malem': 'selamat malam', 'mlm': 'selamat malam',
    'bye': 'sampai jumpa', 'byee': 'sampai jumpa',
    'dadah': 'sampai jumpa', 'dadaa': 'sampai jumpa', 'dad': 'sampai jumpa',
}

CHAR_REPLACEMENTS = {
    '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '7': 't', '8': 'b', '@': 'a',
}

def normalize_repeated_chars(text):
    return re.sub(r'(.)\1{2,}', r'\1', text)

def normalize_slang(text):
    words = text.split()
    normalized = []
    for word in words:
        lower_word = word.lower()
        if lower_word in SLANG_DICTIONARY:
            replacement = SLANG_DICTIONARY[lower_word]
            if replacement:
                normalized.append(replacement)
        else:
            normalized.append(word)
    return ' '.join(normalized)

def normalize_leet(text):
    for char, replacement in CHAR_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    return text

def preprocess_text(text):
    text = text.lower()
    text = normalize_leet(text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = normalize_repeated_chars(text)
    text = normalize_slang(text)
    text = re.sub(r'\s+', ' ', text)
    if USE_STEMMER and stemmer:
        text = stemmer.stem(text)
    return text.strip()

def load_json_dataset(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_similarity(text1, text2):
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

class ChatbotEngine:
    def __init__(self, model_dir='models', dataset_path='datasets.json'):
        self.model_dir = model_dir
        self.dataset_path = dataset_path
        self.all_patterns = []
        self.pattern_to_intent = {}
        self.load_model()

    def preprocess_text(self, text):
        return preprocess_text(text)

    def load_model(self):
        with open(f'{self.model_dir}/vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)

        with open(f'{self.model_dir}/model.pkl', 'rb') as f:
            self.model = pickle.load(f)

        data = load_json_dataset(self.dataset_path)
        self.intent_responses = {}
        for intent in data['intents']:
            tag = intent['tag']
            self.intent_responses[tag] = intent['responses']
            for pattern in intent['patterns']:
                processed = preprocess_text(pattern)
                self.all_patterns.append(processed)
                self.pattern_to_intent[processed] = tag

    def find_similar_intent(self, user_input, threshold=0.6):
        processed_input = preprocess_text(user_input)
        best_match = None
        best_score = 0

        for pattern in self.all_patterns:
            score = calculate_similarity(processed_input, pattern)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = pattern

        if best_match:
            return self.pattern_to_intent[best_match], best_score

        input_words = set(processed_input.split())
        for pattern in self.all_patterns:
            pattern_words = set(pattern.split())
            if input_words and pattern_words:
                overlap = len(input_words & pattern_words)
                total = len(input_words | pattern_words)
                if total > 0:
                    jaccard = overlap / total
                    if jaccard > best_score and jaccard >= 0.4:
                        best_score = jaccard
                        best_match = pattern

        if best_match:
            return self.pattern_to_intent[best_match], best_score

        return None, 0

    def predict_intent(self, user_input, confidence_threshold=0.15):
        processed_input = self.preprocess_text(user_input)
        input_vector = self.vectorizer.transform([processed_input])

        probabilities = self.model.predict_proba(input_vector)[0]
        predicted_class_idx = probabilities.argmax()
        confidence = probabilities[predicted_class_idx]

        predicted_intent = self.model.classes_[predicted_class_idx]

        if confidence < confidence_threshold:
            similar_intent, similarity = self.find_similar_intent(user_input)
            if similar_intent:
                return similar_intent, similarity
            return "unknown", confidence

        return predicted_intent, confidence

    def get_response(self, user_input):
        intent, confidence = self.predict_intent(user_input)

        if intent == "unknown":
            return {
                "intent": "unknown",
                "confidence": float(confidence),
                "response": "Maaf, saya kurang mengerti pertanyaan Anda. Bisa diulang dengan kata-kata yang berbeda? Atau ketik 'bantuan' untuk melihat apa yang bisa saya bantu."
            }

        responses = self.intent_responses.get(intent, ["Maaf, saya tidak memiliki jawaban untuk itu."])
        response_text = random.choice(responses)

        return {
            "intent": intent,
            "confidence": float(confidence),
            "response": response_text
        }
