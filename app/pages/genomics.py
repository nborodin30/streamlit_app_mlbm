import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
from data_utils import parse_fasta, one_hot_encode
import base64

def get_download_link_bytes(bytes_data: bytes, filename: str, mime: str = "application/octet-stream"):
    """
    Возвращает HTML-ссылку (data:URI) для скачивания bytes_data как filename.
    Clicking this link is handled by the browser and does not trigger a Streamlit rerun.
    """
    b64 = base64.b64encode(bytes_data).decode()
    href = f'<a href="data:{mime};base64,{b64}" download="{filename}" ' \
           f'style="display:inline-block;padding:6px 12px;margin:2px;' \
           f'background-color:#4CAF50;color:white;border-radius:4px;' \
           f'text-decoration:none;font-weight:bold;font-family:sans-serif;" ' \
           f'target="_blank" rel="noopener noreferrer">⬇ {filename}</a>'
    return href

def genomics_page(model, device):
    """
    Render the Genomics page for exon detection.

    Args:
        model: Pre-trained ExonIntronCNN model.
        device: Torch device (CPU or CUDA).
    """
    st.header("🧬 Genomics Exon Detection")
    st.write("Upload a FASTA file or enter sequences manually. The model will predict the probability that the center nucleotide belongs to an exon.")

    # FASTA upload
    fasta_file = st.file_uploader("Upload FASTA", type=["fasta", "fa"])
    if fasta_file:
        df = parse_fasta(fasta_file)

        if st.button("Run Prediction on Uploaded FASTA"):
            progress_bar = st.progress(0, text="Processing sequences...")
            total_seqs = len(df)
            X = np.stack([one_hot_encode(seq, 127) for seq in df["sequence"]])
            X = torch.tensor(X).permute(0, 2, 1).to(device)
            with torch.no_grad():
                logits = model(X)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            progress_bar.empty()

            df["Exon_Probability"] = probs
            st.subheader("Prediction Results")
            st.dataframe(df[["id", "sequence", "Exon_Probability"]])
            csv = df.to_csv(index=False).encode("utf-8")
            st.markdown(get_download_link_bytes(csv, "exon_predictions.csv", mime="text/csv"), unsafe_allow_html=True)

            fig_hist = px.histogram(df, x="Exon_Probability", nbins=20, title="Distribution of Exon Probabilities",
                                   color_discrete_sequence=['#1f77b4'])
            st.plotly_chart(fig_hist)

    # Manual input
    with st.form("manual_input_form"):
        manual_input = st.text_area("Enter sequences manually (one per line):")
        submit_manual = st.form_submit_button("Run Prediction on Manual Input")

    if submit_manual:
        lines = [line.strip() for line in manual_input.strip().split("\n") if line.strip()]
        sequences = [line for line in lines if not line.startswith(">")]
        ids = [f"seq_{i+1}" for i in range(len(sequences))]
        df = pd.DataFrame({"id": ids, "sequence": sequences})
        if df.empty or "sequence" not in df.columns or df["sequence"].empty:
            st.error("No valid sequences provided. Please enter at least one sequence of 127 nucleotides containing only A, C, G, T.")
            return

        progress_bar = st.progress(0, text="Processing sequences...")
        total_seqs = len(df)
        X = np.stack([one_hot_encode(seq, 127) for seq in df["sequence"]])
        X = torch.tensor(X).permute(0, 2, 1).to(device)
        with torch.no_grad():
            logits = model(X)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        progress_bar.empty()

        df["Exon_Probability"] = probs
        st.subheader("Prediction Results")
        st.dataframe(df[["id", "sequence", "Exon_Probability"]])
        csv = df.to_csv(index=False).encode("utf-8")
        st.markdown(get_download_link_bytes(csv, "exon_predictions_manual.csv", mime="text/csv"), unsafe_allow_html=True)
        fig_hist = px.histogram(df, x="Exon_Probability", nbins=20, title="Distribution of Exon Probabilities",
                               color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig_hist)