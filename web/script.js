/**
 * Classification de Chiffres Manuscrits - Version Simple
 * Basée sur le modèle PyTorch Quickstart adapté pour MNIST
 */

class DigitClassifier {
    constructor() {
        this.session = null;
        this.canvas = document.getElementById('drawingCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.isDrawing = false;
        this.lastX = 0;
        this.lastY = 0;
        
        this.setupCanvas();
        this.setupEventListeners();
        this.initializeModel();
    }

    setupCanvas() {
        // Configuration simple du canvas
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        this.ctx.strokeStyle = '#000000';
        this.ctx.lineWidth = 12;
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }

    setupEventListeners() {
        // Événements de dessin
        this.canvas.addEventListener('mousedown', this.startDrawing.bind(this));
        this.canvas.addEventListener('mousemove', this.draw.bind(this));
        this.canvas.addEventListener('mouseup', this.stopDrawing.bind(this));
        this.canvas.addEventListener('mouseout', this.stopDrawing.bind(this));

        // Support tactile
        this.canvas.addEventListener('touchstart', this.handleTouch.bind(this));
        this.canvas.addEventListener('touchmove', this.handleTouch.bind(this));
        this.canvas.addEventListener('touchend', this.stopDrawing.bind(this));

        // Boutons
        document.getElementById('clearBtn').addEventListener('click', this.clearCanvas.bind(this));
        document.getElementById('predictBtn').addEventListener('click', this.predict.bind(this));

        // Prédiction automatique
        this.canvas.addEventListener('mouseup', () => {
            setTimeout(() => this.predict(), 500);
        });
    }

    startDrawing(e) {
        this.isDrawing = true;
        const rect = this.canvas.getBoundingClientRect();
        this.lastX = e.clientX - rect.left;
        this.lastY = e.clientY - rect.top;
        
        document.getElementById('instructionText').classList.add('hidden');
    }

    draw(e) {
        if (!this.isDrawing) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const currentX = e.clientX - rect.left;
        const currentY = e.clientY - rect.top;

        this.ctx.beginPath();
        this.ctx.moveTo(this.lastX, this.lastY);
        this.ctx.lineTo(currentX, currentY);
        this.ctx.stroke();

        this.lastX = currentX;
        this.lastY = currentY;
    }

    stopDrawing() {
        this.isDrawing = false;
    }

    handleTouch(e) {
        e.preventDefault();
        const touch = e.touches[0];
        if (!touch) return;
        
        const mouseEvent = new MouseEvent(
            e.type === 'touchstart' ? 'mousedown' : 
            e.type === 'touchmove' ? 'mousemove' : 'mouseup', 
            {
                clientX: touch.clientX,
                clientY: touch.clientY
            }
        );
        this.canvas.dispatchEvent(mouseEvent);
    }

    clearCanvas() {
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        document.getElementById('predictionDisplay').textContent = '?';
        document.getElementById('confidenceDisplay').textContent = 'Confiance: --';
        document.getElementById('instructionText').classList.remove('hidden');
        
        this.updateProbabilityBars(new Array(10).fill(0));
    }

    async initializeModel() {
        try {
            console.log('🔧 Chargement du modèle...');
            
            if (typeof ort === 'undefined') {
                throw new Error('ONNX Runtime non disponible');
            }

            // Configuration ONNX
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/';
            
            // Charger le modèle simple
            this.session = await ort.InferenceSession.create('./digit_model.onnx', {
                executionProviders: ['wasm'],
                logSeverityLevel: 0
            });
            
            console.log('✅ Modèle chargé!');
            console.log('📊 Entrées:', this.session.inputNames);
            console.log('📊 Sorties:', this.session.outputNames);
            
            document.getElementById('loadingOverlay').style.display = 'none';
            this.initializeProbabilityBars();
            
        } catch (error) {
            console.error('❌ Erreur:', error);
            this.showError(error);
        }
    }

    showError(error) {
        const overlay = document.getElementById('loadingOverlay');
        overlay.innerHTML = `
            <div style="color: red; text-align: center; padding: 20px;">
                <h3>❌ Erreur de chargement</h3>
                <p>${error.message}</p>
                <button onclick="location.reload()" 
                        style="padding: 10px 20px; background: #667eea; color: white; 
                               border: none; border-radius: 5px; cursor: pointer; margin-top: 10px;">
                    🔄 Recharger
                </button>
            </div>
        `;
    }

    initializeProbabilityBars() {
        const container = document.getElementById('probabilityBars');
        container.innerHTML = '';
        
        for (let i = 0; i < 10; i++) {
            const barContainer = document.createElement('div');
            barContainer.className = 'probability-bar';
            
            barContainer.innerHTML = `
                <div class="probability-label">${i}</div>
                <div class="probability-fill">
                    <div class="probability-value" style="width: 0%;"></div>
                </div>
            `;
            
            container.appendChild(barContainer);
        }
    }

    updateProbabilityBars(probabilities) {
        const bars = document.querySelectorAll('.probability-value');
        bars.forEach((bar, index) => {
            const percentage = (probabilities[index] * 100).toFixed(1);
            bar.style.width = `${percentage}%`;
            bar.textContent = percentage > 5 ? `${percentage}%` : '';
        });
    }

    preprocessImage() {
        // Redimensionner à 28x28 comme MNIST
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        
        // Fond blanc
        tempCtx.fillStyle = '#ffffff';
        tempCtx.fillRect(0, 0, 28, 28);
        
        // Redimensionner l'image
        tempCtx.drawImage(this.canvas, 0, 0, 28, 28);
        
        // Obtenir les pixels
        const imageData = tempCtx.getImageData(0, 0, 28, 28);
        const data = imageData.data;
        
        // Convertir en format PyTorch (normalisation simple comme ToTensor())
        const input = new Float32Array(1 * 1 * 28 * 28);
        
        for (let i = 0; i < 28 * 28; i++) {
            const pixelIndex = i * 4;
            // Niveau de gris
            const gray = (data[pixelIndex] + data[pixelIndex + 1] + data[pixelIndex + 2]) / 3;
            // Normalisation ToTensor: [0,255] → [0,1], inverser car MNIST: blanc=0, noir=1
            input[i] = (255 - gray) / 255.0;
        }
        
        return input;
    }

    async predict() {
        if (!this.session) {
            console.warn('⚠️ Modèle non chargé');
            return;
        }

        try {
            console.log('🔮 Prédiction...');
            
            const inputData = this.preprocessImage();
            
            // Vérifier que le canvas n'est pas vide
            const sum = inputData.reduce((a, b) => a + b, 0);
            if (sum === 0) {
                console.log('Canvas vide');
                document.getElementById('predictionDisplay').textContent = '?';
                document.getElementById('confidenceDisplay').textContent = 'Dessinez un chiffre';
                this.updateProbabilityBars(new Array(10).fill(0));
                return;
            }
            
            // Créer le tenseur [1, 1, 28, 28]
            const inputTensor = new ort.Tensor('float32', inputData, [1, 1, 28, 28]);
            
            // Inférence
            const results = await this.session.run({ input: inputTensor });
            const logits = Array.from(results.output.data);
            
            // Softmax pour obtenir les probabilités
            const probabilities = this.softmax(logits);
            
            // Prédiction
            const predictedDigit = probabilities.indexOf(Math.max(...probabilities));
            const confidence = probabilities[predictedDigit];
            
            console.log(`🎯 Prédiction: ${predictedDigit} (${(confidence*100).toFixed(1)}%)`);
            
            // Mise à jour de l'interface
            document.getElementById('predictionDisplay').textContent = predictedDigit;
            document.getElementById('confidenceDisplay').textContent = 
                `Confiance: ${(confidence * 100).toFixed(1)}%`;
            
            this.updateProbabilityBars(probabilities);
            
        } catch (error) {
            console.error('❌ Erreur prédiction:', error);
            document.getElementById('predictionDisplay').textContent = '❌';
            document.getElementById('confidenceDisplay').textContent = 'Erreur';
        }
    }

    softmax(logits) {
        const maxLogit = Math.max(...logits);
        const expLogits = logits.map(x => Math.exp(x - maxLogit));
        const sumExp = expLogits.reduce((a, b) => a + b, 0);
        return expLogits.map(x => x / sumExp);
    }
}

// Initialisation
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initialisation...');
    
    if (typeof ort === 'undefined') {
        console.error('❌ ONNX Runtime non disponible');
        document.getElementById('loadingOverlay').innerHTML = `
            <div style="color: red; text-align: center; padding: 20px;">
                <h3>❌ Erreur</h3>
                <p>ONNX Runtime non chargé</p>
            </div>
        `;
        return;
    }
    
    new DigitClassifier();
});