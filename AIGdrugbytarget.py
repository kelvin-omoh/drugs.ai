import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer
from rdkit import Chem
import json

df = pd.read_csv('static/dataset/drug_sequences_target.csv')
df = df[['Sequence', 'target']]
df = df.dropna(subset=['target'])
df.to_csv('static/dataset/drug_sequences_target.csv', index=False)

# extract features from sequences using count amino acids percentage in each sequence

# Import library yang diperlukan
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

"""

# Load dataset senyawa obat
df = pd.read_csv('dataset.csv')

# Fungsi untuk menghitung deskriptor senyawa
def calculate_descriptors(smiles):
    # Konversi smiles ke format molekul RDKit
    mol = Chem.MolFromSmiles(smiles)

    # Hitung deskriptor senyawa
    descriptors = {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'HBA': Lipinski.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol)
    }

    return descriptors

# Hitung deskriptor untuk setiap senyawa dalam dataset
df_descriptors = df['smiles'].apply(calculate_descriptors)

# Fungsi untuk menghitung skor aktivitas biologis
def calculate_activity_score(descriptors):
    # Menggunakan model regresi linier untuk memprediksi skor aktivitas biologis
    # Berdasarkan deskriptor senyawa
    coefficients = np.array([-0.089, -0.698, 0.466, -0.055, -0.337])
    intercept = 3.246
    score = np.dot(coefficients, descriptors) + intercept

    return score

# Hitung skor aktivitas biologis untuk setiap senyawa dalam dataset
df['activity_score'] = df_descriptors.apply(calculate_activity_score, axis=1)

# Urutkan dataset berdasarkan skor aktivitas biologis
df_sorted = df.sort_values('activity_score', ascending=False)

# Tampilkan 10 senyawa dengan skor aktivitas biologis tertinggi
print(df_sorted.head(10))

"""

def calculate_descriptors(seq):
    # Konversi protein sequence ke format molekul RDKit
    mol = Chem.MolFromSequence(seq)

    # Hitung deskriptor senyawa
    descriptors = {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'HBA': Lipinski.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol)
    }

    return descriptors

# fungsi similarity atau kesamaan antara dua text
def similarity(text1, text2):
    # menghitung tfidf
    tfidf = TfidfVectorizer().fit_transform([text1, text2])
    # menghitung similarity
    return ((tfidf * tfidf.T).A)[0, 1]

# Fungsi untuk menghitung skor aktivitas biologis
def calculate_activity_score(descriptors):
    # Mengubah kamus menjadi array numpy
    descriptor_values = np.array(list(descriptors.values()))
    
    # Menggunakan model regresi linier untuk memprediksi skor aktivitas biologis
    # Berdasarkan deskriptor senyawa
    coefficients = np.array([-0.089, -0.698, 0.466, -0.055, -0.337])
    intercept = 3.246
    score = np.dot(coefficients, descriptor_values) + intercept

    return score


class AIGDrug:
    def __init__(self,label):
        self.label = str(label)

    def predict_sequence_class(sequence, model_path='CNNmodel.pkl'):
        # muat model dari file pickle
        with open(model_path, 'rb') as f:
            model = pkl.load(f)

        # lakukan one-hot encoding pada sequence
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        max_len = model.input_shape[1]
        sequence_encoded = np.zeros((max_len, len(amino_acids), 1))
        for i, aa in enumerate(sequence):
            if aa not in amino_acids:
                continue
            else:
                sequence_encoded[i, amino_acids.index(aa)] = 1


        # lakukan prediksi dengan model
        prediction = model.predict(np.expand_dims(sequence_encoded, axis=0))

        # ambil kelas dengan probabilitas tertinggi
        predicted_class = np.argmax(prediction)

        # buka file json yang berisi mapping kelas ke target
        with open('label_mapping.json', 'r') as f:
            target_mapping = json.load(f)

        # ubah key jadi value dan value jadi key
        target_mapping = {v: k for k, v in target_mapping.items()}

        # kembalikan target yang diprediksi
        predicted_class = target_mapping[predicted_class]
        print(predicted_class)

        return predicted_class


    def extract_features(X):
        features = []
        for sequence in X:
            feature = []
            for amino_acid in 'ACDEFGHIKLMNPQRSTVWY':
                feature.append(sequence.count(amino_acid) / len(sequence))
            features.append(feature)
        return features

    def Discriminator(gen):
        gen = gen.upper()
        predicted_class = AIGDrug.predict_sequence_class(gen)
        return predicted_class
    
    def Generate():
        data = pd.read_csv('static/dataset/drug_sequences_target.csv')
        SeqData = data['Sequence']
        gen = []
        # length acak dari dataset
        length = np.random.choice(SeqData.str.len())

        # pilih amino acid berdasarkan panjang sequence sama dengan length, kalo tidak ada yang mendekati length, pilih random
        SeqData = SeqData[SeqData.str.len() == length]
        if len(SeqData) > 0:
            # buat sequence baru dengan mengambil random dari sequence yang sudah ada
            # per residu asam amino nya merupakan pengambilan residu asam amino random dari sequence yang sudah ada
            for i in range(length):
                gen.append(np.random.choice(SeqData.str[i]))
            gen = ''.join(gen)
        else:
            # buat sequence baru dengan mengambil random dari semua asam amino
            gen = ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), length))
        return gen
    
    def GANofVarian(self):
        gen = AIGDrug.Generate()
        y_pred = AIGDrug.Discriminator(gen)
        #print(self.label, end="")
        #print(type(self.label))
        while str(y_pred) != str(self.label):
            gen = AIGDrug.Generate()
            #print(gen, end="")
            y_pred = AIGDrug.Discriminator(gen)
            print(f"{y_pred}")
            print(f"{self.label}")
            # perbandingan label dengan hasil prediksi lakukan similiarity pada hasil prediksi dan label
            sim = similarity(y_pred, self.label)
            print(f"similarity: {sim}")
            if(sim > 0.7):
                break
        #print(gen)
        # evaluasi hasil gen
        Descriptors = calculate_descriptors(gen)
        score = calculate_activity_score(Descriptors)

        # pecah hasil Descriptors menjadi beberapa variabel
        MW = Descriptors['MW']
        LogP = Descriptors['LogP']
        HBD = Descriptors['HBD']
        HBA = Descriptors['HBA']
        TPSA = Descriptors['TPSA']
        return gen, score, MW, LogP, HBD, HBA, TPSA
    
    def getTarget(self):
        return df['target'].unique()
