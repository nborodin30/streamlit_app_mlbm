# ML for Biomedicine Demo Application

This repository contains an interactive **Streamlit** application demonstrating machine learning applications in biomedicine.  
The application integrates **genomic data analysis** (exon detection) and **natural language processing** (disease prediction from symptoms).

---

## 📌 Overview
The app provides an educational demonstration of end-to-end ML pipelines in biomedical contexts.  
It consists of four main modules:

1. **Home Page** – Application overview and demo video.  
2. **Exploratory Data Analysis** – Genomic sequence statistics with visualization.  
3. **Genomics Exon Detection** – CNN-based exon/intron classification.  
4. **Natural Language Processing** – Disease prediction from symptom descriptions.  

The backend includes a complete training pipeline for the genomic CNN model, implemented with **Hydra** for configuration management.

---

## ✨ Features

### Genomic Exploratory Data Analysis
- Upload FASTA files containing nucleotide sequences and binary masks (0 = intron, 1 = exon).
- Compute class distributions and chromosome-level statistics.
- Visualize sequence length distributions and exon ratios per chromosome.
- Validate alignment between sequence and mask IDs.

### Exon/Intron Classification
- Pre-trained **1D CNN** model for predicting exon probability of center nucleotides (127-bp windows).  
- Supports:
  - **Batch FASTA processing**.
  - **Manual sequence input**.  
- One-hot encoding of nucleotides (A, C, G, T, N).  
- Export results as CSV.  
- Interactive histograms of predictions via **Plotly**.

### NLP Pipeline for Symptom-Based Disease Prediction
- Upload CSV datasets (symptoms + disease labels).  
- Dynamic column selection for text and labels.  
- Multiple text representations: **TF-IDF** or **Sentence Transformer embeddings**.  
- 5 classifiers with cross-validation: Logistic Regression, SVM, Random Forest, KNN.  
- Metrics: Accuracy, Precision, Recall, F1-score, **Matthews Correlation Coefficient (MCC)**.  
- Word frequency analysis, histograms, word clouds.  
- Export trained models (pickle, joblib, numpy).

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+  
- Conda or Mamba  
- Git  

### Setup
```bash
# Clone repository
git clone https://github.com/nborodin30/streamlit_app_mlbm.git
cd streamlit_app_mlbm

# Create and activate conda environment
conda env create -f environment.yml
conda activate streamlit_app_mlbm
```

---

## ▶️ Usage
Run the application:
```bash
streamlit run app/streamlit_app.py
```

The app will be available at [http://localhost:8501](http://localhost:8501).

---


## 📊 Technical Details

### CNN for Exon Prediction
- Input: `(batch_size, 5, 127)` (one-hot encoding).  
- Conv1D + ReLU + Dropout blocks.  
- Global average pooling.  
- Linear output layer → intron/exon classes.  
- Loss: cross-entropy with class weights.  
- Evaluation: MCC, ROC, PR curves.  

### Training Pipeline
- **Data**: FASTA parsing via Biopython, sliding windows, chromosome-based splits.  
- **Features**: Balancing, early stopping, checkpoints, logging.  
- **Config management**: Hydra.  

### NLP Pipeline
- Stopword removal, tokenization, filtering.  
- Representations: TF-IDF (max 5000 features) or pretrained Sentence Transformers.  
- 5-fold CV with multiple classifiers.  
- Weighted multi-class metrics, with standard deviation across folds.  
- Export trained models + vectorizers.  

---

## 📦 Sample Data
Sample datasets (provided in `data/`):
- `hg38_10pct.fasta` – Reference human genome subset.  
- `mask_10pct.fasta` – Binary exon/intron mask.  
- `Symptoms2Diseases.csv` – Symptoms → disease labels.  

Users can also upload custom datasets via the app.

---

## 🧾 Dependencies
Main packages:
- **Web UI**: `streamlit`  
- **ML/DL**: `torch`, `scikit-learn`  
- **NLP**: `spacy`, `sentence-transformers`, `nltk`  
- **Bioinformatics**: `biopython`  
- **Visualization**: `plotly`, `matplotlib`, `seaborn`, `wordcloud`  
- **Config**: `hydra-core`, `pyyaml`  

See `environment.yml` for exact versions.

---

## ✅ Quality Checklist
- [x] No data leakage; preprocessing documented.  
- [x] At least 2 model families compared.  
- [x] Metrics: MCC, confusion matrix, ROC/PR.  
- [x] Reproducibility: `environment.yml`, fixed seeds.  
- [x] Clean UI/UX with error handling and progress bars.  

---

## 🙏 Acknowledgments
- Streamlit team – web framework.  
- PyTorch developers – DL backend.  
- SpaCy & Hugging Face – NLP tools.  
- Biopython – genomics utilities.  
- Hydra – configuration management.  
