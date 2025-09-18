import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
import streamlit as st
from config import load_config
from model_utils import load_model, load_spacy_model
from pages.home import home_page
from pages.eda import eda_page
from pages.genomics import genomics_page
from pages.nlp import nlp_page

# Set page configuration
st.set_page_config(page_title="ML for Biomedicine", layout="wide")

# Load configuration and models
config = load_config()
model, device = load_model(config)
nlp_spacy = load_spacy_model()

# Define tabs
tabs = st.tabs(["Home", "Exploratory Data Analysis", "Genomics Exon Detection", "Natural Language Processing"])
with tabs[0]:
    home_page()
with tabs[1]:
    eda_page()
with tabs[2]:
    genomics_page(model, device)
with tabs[3]:
    nlp_page(nlp_spacy)