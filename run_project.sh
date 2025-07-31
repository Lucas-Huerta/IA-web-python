#!/bin/bash

echo "🚀 === Configuration du projet MNIST Classification ==="

# Vérifications initiales
if [ ! -d ".venv" ]; then
    echo "❌ Erreur: L'environnement virtuel .venv n'existe pas"
    echo "💡 Créez d'abord un environnement virtuel avec: python -m venv .venv"
    exit 1
fi

# Activation de l'environnement virtuel
echo "🔧 Activation de l'environnement virtuel..."
source .venv/bin/activate

if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Erreur: Impossible d'activer l'environnement virtuel"
    exit 1
fi

echo "✅ Environnement virtuel activé: $VIRTUAL_ENV"

# Installation des dépendances
echo "📦 Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

# Création du dossier web
mkdir -p web

# Vérification des fichiers web
if [ ! -f "web/index.html" ]; then
    echo "❌ Erreur: Les fichiers web n'existent pas."
    echo "💡 Assurez-vous que index.html, style.css et script.js sont dans le dossier web/"
    exit 1
fi

# Entraînement du modèle
echo "🧠 Lancement de l'entraînement du modèle MNIST..."
python train_mnist.py

# Vérification de la création du modèle
if [ ! -f "web/mnist_model.onnx" ]; then
    echo "❌ Erreur: Le modèle ONNX n'a pas été créé"
    exit 1
fi

echo "✅ Modèle créé avec succès!"

# Démarrage du serveur web
echo ""
echo "🌐 Démarrage du serveur web..."
echo "🔗 Ouvrez votre navigateur à l'adresse: http://localhost:8000"
echo "🔗 Page de debug disponible à: http://localhost:8000/debug.html"
echo "⏹️  Appuyez sur Ctrl+C pour arrêter le serveur"
echo ""

cd web && python -m http.server 8000 