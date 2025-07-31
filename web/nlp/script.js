class PoetryGenerator {
    constructor() {
        this.isGenerating = false;
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Bouton de génération
        const generateBtn = document.getElementById('generateBtn');
        if (generateBtn) {
            generateBtn.addEventListener('click', () => {
                this.generatePoem();
            });
        }

        // Entrée sur l'input
        const startWordInput = document.getElementById('startWord');
        if (startWordInput) {
            startWordInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.generatePoem();
                }
            });
        }
    }

    async generatePoem() {
        if (this.isGenerating) return;

        const startWordInput = document.getElementById('startWord');
        if (!startWordInput) {
            console.error('Element startWord non trouvé');
            return;
        }

        const startWord = startWordInput.value.trim();
        
        if (!startWord) {
            alert('Veuillez entrer un mot pour commencer le poème');
            return;
        }

        this.isGenerating = true;
        this.showLoading();
        
        try {
            // Appeler l'API du modèle PyTorch
            const response = await this.callPoetryAPI(startWord);
            if (response.success) {
                this.displayPoem(response.poem);
            } else {
                throw new Error(response.error);
            }
            
        } catch (error) {
            console.error('Erreur génération:', error);
            // Fallback vers simulation
            this.simulatePoem(startWord);
        } finally {
            this.hideLoading();
            this.isGenerating = false;
        }
    }

    async callPoetryAPI(startWord) {
        const response = await fetch('http://localhost:5000/api/generate-poem', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                start_word: startWord,
                length: 200,
                temperature: 0.8
            })
        });

        if (!response.ok) {
            throw new Error('Erreur API');
        }

        return await response.json();
    }

    simulatePoem(startWord) {
        // Simulation de fallback
        const poemTemplates = {
            'amour': `${startWord}, sentiment profond qui traverse les âges,
comme une mélodie douce dans le cœur sauvage.
les mots dansent et s'entrelacent,
créant des vers où l'émotion se trace.

dans le silence de la nuit étoilée,
les rêves prennent leur envol ailé.
chaque syllabe porte en elle
la beauté d'une vie nouvelle.`,

            'nature': `${startWord}, mère nourricière aux mille visages,
tes forêts murmurent d'anciens messages.
les rivières chantent leur course éternelle,
portant l'espoir d'une terre nouvelle.

sous tes arbres aux branches étendues,
l'âme humaine trouve ses vérités perdues.
chaque feuille qui danse dans le vent
raconte l'histoire du temps.`,

            'rêve': `${startWord} doré aux couleurs de l'infini,
tu portes l'espoir d'un monde béni.
dans les méandres de la nuit profonde,
tu tisses des histoires vagabondes.

tes images flottent comme des nuages,
peignant dans l'âme de doux paysages.
entre réel et imaginaire,
tu révèles l'extraordinaire.`
        };

        const selectedTemplate = poemTemplates[startWord.toLowerCase()] || 
                                 this.generateDefaultPoem(startWord);
        
        setTimeout(() => {
            this.displayPoem(selectedTemplate);
            this.hideLoading();
            this.isGenerating = false;
        }, 2000);
    }

    generateDefaultPoem(word) {
        return `${word}, mot mystérieux qui éveille l'imaginaire,
comme une clé ouvrant les portes de l'ordinaire.
dans le jardin secret des mots choisis,
naît un poème aux couleurs infinies.

les vers s'échappent tel un fleuve de pensées,
créant des images aux reflets nacrés.
entre les lignes se cache la magie
d'une âme qui rêve et qui s'épanouit.

${word}, tu inspires le poète,
guidant sa plume vers des contrées secrètes.
dans chaque syllabe résonne
l'écho d'une mélodie qui frissonne.`;
    }

    displayPoem(poem) {
        const outputDiv = document.getElementById('poemOutput');
        if (!outputDiv) {
            console.error('Element poemOutput non trouvé');
            return;
        }

        outputDiv.innerHTML = poem;
        outputDiv.classList.add('has-content');
        
        // Animation d'apparition
        outputDiv.style.opacity = '0';
        setTimeout(() => {
            outputDiv.style.transition = 'opacity 0.8s ease-in-out';
            outputDiv.style.opacity = '1';
        }, 100);
    }

    showLoading() {
        const button = document.getElementById('generateBtn');
        if (button) {
            button.innerHTML = '<span class="loading"></span>Génération en cours...';
            button.disabled = true;
        }
        
        const outputDiv = document.getElementById('poemOutput');
        if (outputDiv) {
            outputDiv.innerHTML = '<p class="placeholder">Création de votre poème...</p>';
            outputDiv.classList.remove('has-content');
        }
    }

    hideLoading() {
        const button = document.getElementById('generateBtn');
        if (button) {
            button.innerHTML = 'Générer un Poème';
            button.disabled = false;
        }
    }
}

// Initialiser l'application quand le DOM est chargé
document.addEventListener('DOMContentLoaded', () => {
    console.log('🎭 Générateur de Poèmes initialisé');
    new PoetryGenerator();
    
    // Ajouter suggestions de mots
    const input = document.getElementById('startWord');
    if (input) {
        const wordSuggestions = [
            'amour', 'nature', 'rêve', 'espoir', 'liberté', 
            'océan', 'étoile', 'bonheur', 'mélancolie', 'voyage'
        ];
        
        input.addEventListener('focus', () => {
            if (!input.value) {
                const randomWord = wordSuggestions[Math.floor(Math.random() * wordSuggestions.length)];
                input.placeholder = `Ex: ${randomWord}`;
            }
        });
    }
});