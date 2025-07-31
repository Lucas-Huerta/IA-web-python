"""
Serveur API Flask pour la génération de poèmes en temps réel
Charge le modèle PyTorch entraîné et génère des poèmes via API REST
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn.functional as F
from torch import nn
import os

app = Flask(__name__)
CORS(app)

# Configuration du device
device = (
    "mps" 
    if torch.backends.mps.is_available() 
    else "cpu"
)

# Variables globales pour le modèle
model = None
char_to_idx = None
idx_to_char = None
vocab_size = None

class CharRNN(nn.Module):
    """Même architecture que dans train_nlp.py"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(CharRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden
    
    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

def load_model():
    """Charge le modèle entraîné"""
    global model, char_to_idx, idx_to_char, vocab_size
    
    try:
        # Charger le modèle sauvegardé
        checkpoint = torch.load('poetry_generator.pth', map_location=device)
        
        char_to_idx = checkpoint['char_to_idx']
        idx_to_char = checkpoint['idx_to_char']
        vocab_size = checkpoint['vocab_size']
        
        # Recréer le modèle avec les mêmes paramètres
        model = CharRNN(vocab_size, 128, 256, 2).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Modèle chargé: {vocab_size} caractères dans le vocabulaire")
        return True
        
    except FileNotFoundError:
        print("Modèle non trouvé. Lancez d'abord train_nlp.py pour l'entraîner")
        return False
    except Exception as e:
        print(f"Erreur chargement modèle: {e}")
        return False

def generate_poem_with_model(start_word, length=200, temperature=0.8):
    """Génère un poème avec le modèle PyTorch"""
    if model is None:
        return "Modèle non chargé. Entraînez d'abord le modèle avec train_nlp.py"
    
    model.eval()
    
    # Convertir le mot de départ en indices
    start_word = start_word.lower()
    chars = [char_to_idx.get(c, 0) for c in start_word]
    input_seq = torch.tensor(chars, dtype=torch.long).unsqueeze(0).to(device)
    
    generated = start_word
    hidden = model.init_hidden(1)
    
    # Générer caractère par caractère
    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            
            # Appliquer température
            logits = output[0, -1] / temperature
            probabilities = F.softmax(logits, dim=0)
            
            # Échantillonner
            next_char_idx = torch.multinomial(probabilities, 1).item()
            next_char = idx_to_char[next_char_idx]
            
            generated += next_char
            
            # Arrêter si on trouve plusieurs sauts de ligne
            if generated.count('\n') >= 8:
                break
                
            input_seq = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)
    
    return generated

@app.route('/')
def index():
    """Servir la page web"""
    return app.send_static_file('nlp/index.html')

@app.route('/nlp/')
def nlp_index():
    """Route alternative pour la page NLP"""
    return app.send_static_file('nlp/index.html')

@app.route('/api/generate-poem', methods=['POST'])
def generate_poem_api():
    """API pour générer un poème"""
    try:
        data = request.json
        start_word = data.get('start_word', 'poésie')
        length = min(data.get('length', 200), 300)  # Limiter la longueur
        temperature = data.get('temperature', 0.8)
        
        # Générer le poème
        poem = generate_poem_with_model(start_word, length, temperature)
        
        return jsonify({
            'success': True,
            'poem': poem,
            'start_word': start_word
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/status')
def status():
    """Vérifier le statut du modèle"""
    return jsonify({
        'model_loaded': model is not None,
        'vocab_size': vocab_size if vocab_size else 0
    })

if __name__ == '__main__':
    print("=== Serveur API Générateur de Poèmes ===")
    
    # Charger le modèle au démarrage
    if load_model():
        print("Serveur prêt à générer des poèmes!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Impossible de démarrer sans modèle entraîné")
        print("Lancez d'abord: python train_nlp.py") 