import torch
import spacy
import streamlit as st
from ml.model import ExonIntronCNN

@st.cache_resource
def load_model(config):
    """
    Load the pre-trained ExonIntronCNN model.

    Args:
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        tuple: (model, device) where model is the loaded ExonIntronCNN and device is the torch device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config["model"]
    
    model = ExonIntronCNN(
        in_channels=5,  # Nucleotides A, C, G, T, N
        conv_channels=model_cfg["conv_channels"],
        kernel_size=model_cfg["kernel_size"],
        dropout=model_cfg["dropout"],
        num_classes=2  # Binary classification: exon/intron
    )
    
    checkpoint = torch.load("outputs/2025-09-17-20-16-33/model.pt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, device

@st.cache_resource
def load_spacy_model():
    """
    Load the SpaCy English model.

    Returns:
        spacy.language.Language: Loaded SpaCy model or None if not found.
    """
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.error("SpaCy model 'en_core_web_sm' not found. Please install it by running: `python -m spacy download en_core_web_sm`")
        return None