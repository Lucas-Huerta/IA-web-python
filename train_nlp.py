"""
Script d'entraînement NLP - Génération de texte et analyse
Exploration des différents encodages et modèles récurrents
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import re
from collections import Counter
import os

print("=== Projet NLP - Génération de texte ===")

# CONFIG GÉNÉRALE
device = (
    "mps" 
    if torch.backends.mps.is_available() 
    else "cpu"
)
print(f"Using {device} device")


# TODO changer un peu les valeurs pour vérfier les résultats obtenu
BATCH_SIZE = 64 # avant 32
SEQUENCE_LENGTH = 100  # séquences d'entraînement => nombre de caractères ======== avant 50
EMBEDDING_DIM = 128   # Dimension des embeddings => mémoire
HIDDEN_DIM = 256      # Dimension cachée des RNN => nuance des mots 
NUM_LAYERS = 2        # Nombre de couches RNN => couche = profondeur
LEARNING_RATE = 0.001
NUM_EPOCHS = 10 #avant 10

# PRÉPARATION DES DONNÉES ET ENCODAGE

class TextDataset(Dataset):
    """Dataset personnalisé pour le texte au niveau caractère"""
    def __init__(self, text, seq_length, char_to_idx, idx_to_char):
        self.text = text
        self.seq_length = seq_length
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        
    def __len__(self):
        return len(self.text) - self.seq_length
    
    def __getitem__(self, idx):
        # Extraire séquence d'entrée et cible (décalée d'un caractère)
        input_seq = self.text[idx:idx + self.seq_length]
        target_seq = self.text[idx + 1:idx + self.seq_length + 1]
        
        # Convertir caractères en indices
        input_indices = [self.char_to_idx[char] for char in input_seq]
        target_indices = [self.char_to_idx[char] for char in target_seq]
        
        return torch.tensor(input_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)

def load_and_preprocess_text(dataset_source="huggingface_poems"):
    """Charge le dataset de poèmes depuis Hugging Face"""
    if dataset_source == "huggingface_poems":
        return load_huggingface_poems_dataset()
    else:
        return load_fallback_text()

def load_huggingface_poems_dataset():
    try:
        from datasets import load_dataset
        
        dataset = load_dataset("checkai/instruction-poems")
        
        # Prendre le premier split disponible
        if 'train' in dataset:
            data = dataset['train']
        else:
            data = dataset[list(dataset.keys())[0]]
        
        # Chercher la colonne de texte
        possible_columns = ['text', 'poem', 'content', 'instruction', 'output', 'response']
        text_column = None
        
        for col in possible_columns:
            if col in data.features:
                text_column = col
                break
        
        if text_column is None:
            for col, feature in data.features.items():
                if feature.dtype == 'string':
                    text_column = col
                    break
        
        if text_column is None:
            return load_fallback_text()
        
        # Extraire (limité à 500 pour rapidité)
        texts = []
        for i, example in enumerate(data):
            if i >= 500:
                break
            text = example[text_column]
            if text and len(str(text).strip()) > 50:
                texts.append(str(text).strip())
        
        if not texts:
            return load_fallback_text()
        
        combined_text = '\n\n'.join(texts)
        return clean_text(combined_text)
        
    except Exception as e:
        return load_fallback_text()

def clean_text(text):
    text = text.lower()
    # lettres, espaces et ponctuation
    text = re.sub(r'[^a-zA-Zàâäéèêëïîôùûüÿç\s.,!?\n\']', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)  
    return text.strip()

def load_fallback_text():
    """Corpus de fallback"""
    text = """
    les sanglots longs des violons de l'automne blessent mon cœur d'une langueur monotone.
    tout suffocant et blême, quand sonne l'heure, je me souviens des jours anciens et je pleure.
    et je m'en vais au vent mauvais qui m'emporte deçà, delà, pareil à la feuille morte.
    
    il pleure dans mon cœur comme il pleut sur la ville, quelle est cette langueur qui pénètre mon cœur?
    ô bruit doux de la pluie par terre et sur les toits! pour un cœur qui s'ennuie ô le chant de la pluie!
    
    dans le vieux parc solitaire et glacé deux formes ont tout à l'heure passé.
    leurs yeux sont morts et leurs lèvres sont molles, et l'on entend à peine leurs paroles.
    
    le ciel est, par-dessus le toit, si bleu, si calme! un arbre, par-dessus le toit, berce sa palme.
    la cloche, dans le ciel qu'on voit, doucement tinte. un oiseau sur l'arbre qu'on voit chante sa plainte.
    """
    return clean_text(text)

def create_char_mappings(text):
    """Crée les dictionnaires caractère ↔ index"""
    chars = sorted(list(set(text)))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    return char_to_idx, idx_to_char

# MODÈLES RÉCURRENTS

class CharRNN(nn.Module):
    """Modèle RNN simple pour génération de texte au niveau caractère"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(CharRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Couches du modèle
        self.embedding = nn.Embedding(vocab_size, embedding_dim)      # Embedding des caractères
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)  # LSTM pour mémoire
        self.fc = nn.Linear(hidden_dim, vocab_size)                   # Couche finale pour prédiction
        
    def forward(self, x, hidden=None):
        # x: (batch_size, seq_length)
        embedded = self.embedding(x)           # (batch_size, seq_length, embedding_dim)
        output, hidden = self.rnn(embedded, hidden)  # (batch_size, seq_length, hidden_dim)
        output = self.fc(output)               # (batch_size, seq_length, vocab_size)
        return output, hidden
    
    def init_hidden(self, batch_size):
        """Initialise l'état caché du LSTM"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

# FONCTIONS D'ENTRAÎNEMENT

def train_model(model, dataloader, criterion, optimizer, num_epochs):
    """Entraîne le modèle de génération de texte"""
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            # move data to device
            data, target = data.to(device), target.to(device)
            
            # reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = model(data)
            
            # Calculer loss (reshape pour CrossEntropyLoss)
            loss = criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Afficher progression tous les 2 epochs
        if (epoch + 1) % 2 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}')

def generate_text(model, start_string, length, char_to_idx, idx_to_char, temperature=1.0):
    """Génère du texte à partir d'un mot de départ"""
    model.eval()
    
    # Convertir le mot de départ en indices
    chars = [char_to_idx.get(c, 0) for c in start_string.lower()]
    input_seq = torch.tensor(chars, dtype=torch.long).unsqueeze(0).to(device)
    
    generated = start_string
    hidden = model.init_hidden(1)
    
    # Générer caractère par caractère
    with torch.no_grad():
        for _ in range(length):
            # Prédire le prochain caractère
            output, hidden = model(input_seq, hidden)
            
            # Appliquer température pour contrôler créativité
            logits = output[0, -1] / temperature
            probabilities = F.softmax(logits, dim=0)
            
            # Échantillonner le prochain caractère
            next_char_idx = torch.multinomial(probabilities, 1).item()
            next_char = idx_to_char[next_char_idx]
            
            generated += next_char
            
            # Préparer pour prochaine itération
            input_seq = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)
    
    return generated

def main():
    """Pipeline principal d'exécution"""
    
    print("=== Entraînement du générateur de poèmes ===")
    
    # 1. Charger et préparer les données
    text = load_and_preprocess_text("huggingface_poems")
    char_to_idx, idx_to_char = create_char_mappings(text)
    vocab_size = len(char_to_idx)
    
    print(f"Corpus chargé: {len(text)} caractères, vocabulaire: {vocab_size}")
    
    # 2. Créer dataset et dataloader
    dataset = TextDataset(text, SEQUENCE_LENGTH, char_to_idx, idx_to_char)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Initialiser le modèle
    model = CharRNN(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Entraîner le modèle
    print("Début de l'entraînement...")
    train_model(model, dataloader, criterion, optimizer, NUM_EPOCHS)
    
    # 5. Tester la génération
    test_word = "amour"
    generated_poem = generate_text(model, test_word, 200, char_to_idx, idx_to_char, temperature=0.8)
    print(f"\nPoème généré à partir de '{test_word}':")
    print(generated_poem)
    
    # 6. Sauvegarder le modèle
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab_size': vocab_size
    }, 'poetry_generator.pth')
    
    print("\nModèle sauvegardé: poetry_generator.pth")

if __name__ == "__main__":
    main()
