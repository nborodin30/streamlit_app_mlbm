import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import parse_fasta
import numpy as np
import plotly.express as px

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
        
        # Determine the top chromosomes to display
        top_n = 10
        top_chromosomes = per_chr["total_len"].nlargest(top_n).index
        per_chr_top = per_chr.loc[top_chromosomes]

        # Calculate exon ratio for remaining chromosomes
        remaining_chromosomes = per_chr.index.difference(top_chromosomes)
        if not remaining_chromosomes.empty:
            other_total_len = per_chr.loc[remaining_chromosomes, "total_len"].sum()
            other_ones = per_chr.loc[remaining_chromosomes, "ones"].sum()
            other_exon_ratio = other_ones / other_total_len if other_total_len > 0 else 0
            
            other_row = pd.DataFrame({
                "total_len": [other_total_len],
                "ones": [other_ones],
                "zeros": [other_total_len - other_ones],
                "exon_ratio": [other_exon_ratio]
            }, index=["Other"])
            per_chr_top = pd.concat([per_chr_top, other_row])
        
        # Create a boxplot for sequence length distribution by chromosome
        df_for_plot = df.copy()
        df_for_plot['chr_group'] = df_for_plot['chr'].apply(lambda x: x if x in top_chromosomes else 'Other')

        fig1 = px.box(df_for_plot, x="chr_group", y="seq_len", 
                     title="Sequence Length Distribution by Chromosome",
                     labels={'chr_group': 'Chromosome Group', 'seq_len': 'Sequence Length'})
        fig1.update_layout(xaxis_title="Chromosome Group", yaxis_title="Sequence Length")
        st.plotly_chart(fig1)

        # Plot 2: Exon Ratio per Chromosome (Grouped)
        fig, ax = plt.subplots(figsize=(16, 6))
        per_chr_top_sorted = per_chr_top.sort_values("exon_ratio", ascending=False)
        sns.barplot(x=per_chr_top_sorted.index, y=per_chr_top_sorted["exon_ratio"], ax=ax)
        ax.set_title("Exon Ratio per Chromosome (Top 10 + Other)")
        ax.set_ylabel("Exon Ratio")
        ax.set_xlabel("Chromosome")
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        st.pyplot(fig)
