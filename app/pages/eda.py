import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import parse_fasta

def eda_page():
    """
    Render the Exploratory Data Analysis (EDA) page for genomic data.
    """
    st.header("🔍 Genomic EDA: Sequences + Mask")
    st.write("Upload two FASTA files: one with nucleotide sequences and one with mask (0/1) sequences.")

    seq_file = st.file_uploader("Upload sequence FASTA", type=["fasta"])
    mask_file = st.file_uploader("Upload mask FASTA", type=["fasta"])

    if seq_file and mask_file:
        st.info("Parsing files...")
        seq_df = parse_fasta(seq_file)
        mask_df = parse_fasta(mask_file)

        if not seq_df["id"].equals(mask_df["id"]):
            st.error("Sequence IDs in the two FASTA files do not match!")
            return

        df = pd.DataFrame()
        df["id"] = seq_df["id"]
        df["sequence"] = seq_df["sequence"]
        df["mask"] = mask_df["sequence"].apply(lambda x: [int(b) for b in x])
        df["seq_len"] = df["sequence"].apply(len)
        df["mask_ones"] = df["mask"].apply(sum)
        df["chr"] = df["id"].apply(lambda x: x.split(":")[0] if ":" in x else x)

        # Overall base-level distribution
        st.subheader("Base-level Class Distribution (Overall)")
        total_ones = df["mask_ones"].sum()
        total_bases = df["seq_len"].sum()
        total_zeros = total_bases - total_ones
        st.write(f"**Overall bases:** {total_bases:,} | **exon (1):** {total_ones:,} ({total_ones/total_bases:.2%}) | **intron (0):** {total_zeros:,} ({total_zeros/total_bases:.2%})")

        # Per-chromosome distribution
        st.subheader("Per-chromosome Base-level Distribution")
        per_chr = df.groupby("chr").agg(total_len=("seq_len","sum"), ones=("mask_ones","sum"))
        per_chr["zeros"] = per_chr["total_len"] - per_chr["ones"]
        per_chr["exon_ratio"] = per_chr["ones"] / per_chr["total_len"]
        st.dataframe(per_chr.sort_index())

        # Plots
        st.subheader("Plots")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df["seq_len"], bins=30, kde=True, ax=ax[0])
        ax[0].set_title("Sequence Length Distribution")
        ax[0].set_xlabel("Length")

        per_chr_sorted = per_chr.sort_index()
        sns.barplot(x=per_chr_sorted.index, y=per_chr_sorted["exon_ratio"], ax=ax[1])
        ax[1].set_title("Exon Ratio per Chromosome")
        ax[1].set_ylabel("Exon Ratio")
        ax[1].set_xlabel("Chromosome")
        plt.xticks(rotation=45)
        st.pyplot(fig)