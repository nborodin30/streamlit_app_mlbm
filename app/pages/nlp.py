import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
from wordcloud import WordCloud
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import joblib
import pickle
import io
import seaborn as sns  
import random
import time
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import nltk
from PIL import Image

import base64

nltk.download('stopwords')

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

def nlp_page(nlp_spacy):
    """
    Render the NLP page for disease prediction from text data.

    Args:
        nlp_spacy: Loaded SpaCy model for text processing.
    """
    # Initialize session_state to preserve CV results and state across reruns
    if 'cv_results_df' not in st.session_state:
        st.session_state.cv_results_df = None
    if 'cv_complete' not in st.session_state:
        st.session_state.cv_complete = False
    if 'symptoms' not in st.session_state:
        st.session_state.symptoms = []
    if 'diseases' not in st.session_state:
        st.session_state.diseases = []




    st.header("🧠 NLP: Disease Prediction")
    
    # File Upload or Preloaded Dataset
    df = pd.DataFrame()
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if st.session_state.cv_complete:
            st.session_state.cv_complete = False
            st.session_state.cv_results_df = None
            st.session_state.symptoms = []
            st.session_state.diseases = []
    else:
        try:
            df = pd.read_csv("data/Symptoms2Diseases.csv")
            st.info("Using preloaded Symptoms2Diseases.csv dataset.")
        except FileNotFoundError:
            st.error("Dataset file not found. Please upload a CSV file.")
            return

    # Dataset Preview
    st.subheader("Dataset Preview")
    rows_to_display = st.number_input(
        "Enter number of rows to display",
        min_value=1,
        max_value=len(df),
        value=min(10, len(df)),
        step=1
    )
    st.dataframe(df.head(rows_to_display))

    # Column Selection
    cols = df.columns.tolist()

    default_text_col = next((c for c in cols if c.lower() == 'text'), 
                           next((c for c in cols if c.lower() == 'symptom'), cols[0]))
    default_label_col = next((c for c in cols if c.lower() == 'label'), 
                            next((c for c in cols if c.lower() == 'disease'), cols[0]))

    # Always show selectbox for column selection with defaults
    st.subheader("Column Selection")
    text_col = st.selectbox(
        "Select Text Column (e.g., text or symptom)",
        cols,
        index=cols.index(default_text_col),
        key="text_col"
    )
    label_col = st.selectbox(
        "Select Label Column (e.g., label or disease)",
        cols,
        index=cols.index(default_label_col),
        key="label_col"
    )
     # Verify selected columns exist
    if text_col not in df.columns or label_col not in df.columns:
        st.error(f"Selected columns (text: {text_col}, label: {label_col}) not found in the dataset.")
        return

    try:
        df = df[[text_col, label_col]].dropna()
    except KeyError:
        st.error("Error accessing selected columns. Please ensure the columns contain valid data.")
        return
    
    symptoms = df[text_col].astype(str).tolist()
    diseases = df[label_col].astype(str).tolist()
    
    unique_labels = sorted(set(diseases))
    st.write(f"Unique Labels: {len(unique_labels)} - {', '.join(unique_labels[:5]) + '...' if len(unique_labels) > 5 else ', '.join(unique_labels)}")

    if nlp_spacy is None:
        return

    # Exploratory Data Analysis
    st.subheader("Exploratory Data Analysis")
    text_corpus = " ".join(symptoms)
    spacy_stopwords = nlp_spacy.Defaults.stop_words
    stop_words = set(stopwords.words('english')).union(spacy_stopwords)
    words = re.findall(r'\w+', text_corpus.lower())
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    word_freq = Counter(filtered_words)

    # Top N Frequent Words
    st.subheader("Top Frequent Words")
    top_n = st.slider("Select number of top words to display", min_value=5, max_value=50, value=10, step=1)
    top_words = word_freq.most_common(top_n)
    top_words_dict = dict(word_freq.most_common(100))
    top_df = pd.DataFrame(top_words, columns=['Word', 'Count'])
    st.table(top_df)


    # Dynamic Histogram for Top Words
    st.subheader("Frequency Histogram of Top Words")
    words_list, counts_list = zip(*top_words)  # Unzip the top words and counts
    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(words_list), y=list(counts_list), ax=ax_hist)
    ax_hist.set_xlabel("Words")
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title(f"Top {top_n} Most Frequent Words")
    plt.tight_layout()
    st.pyplot(fig_hist)
    plt.close(fig_hist)

    # Dynamic WordCloud
    st.subheader("Dynamic WordCloud")
    if 'animation_ran' not in st.session_state:
        st.session_state.animation_ran = False
    if 'run_animation_now' not in st.session_state:
        st.session_state.run_animation_now = False

    if st.button('Reload', key='reload_wordcloud'):
        st.session_state.run_animation_now = True

    if not st.session_state.animation_ran:
        st.session_state.run_animation_now = True

    if st.session_state.run_animation_now:
        wordcloud_placeholder = st.empty()
        colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
        start_time = time.time()
        animation_duration = 2

        while (time.time() - start_time) < animation_duration:
            random_colormap = random.choice(colormaps)
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap=random_colormap,
                prefer_horizontal=0.8,
                stopwords=set()
            ).generate_from_frequencies(top_words_dict)
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            wordcloud_placeholder.pyplot(fig_wc)
            time.sleep(2)
            plt.close(fig_wc)

        st.session_state.animation_ran = True
        st.session_state.run_animation_now = False
        st.rerun()

    if st.session_state.animation_ran:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            prefer_horizontal=0.8,
            stopwords=set()
        ).generate_from_frequencies(top_words_dict)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
        plt.close(fig_wc)

        img_buffer = io.BytesIO()
        wordcloud.to_image().save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()

        # Генерация ссылки для скачивания без ререндер
        st.markdown(get_download_link_bytes(img_bytes, "wordcloud.png", mime="image/png"),
                    unsafe_allow_html=True)
    # Model Training
    @st.cache_data
    def get_embedder(model_name):
        return SentenceTransformer(model_name)

    if 'classifiers_templates' not in st.session_state:
        st.session_state.classifiers_templates = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "SVC": SVC(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "KNN": KNeighborsClassifier()
        }

    st.subheader("Automated Training Pipeline")

    
    rep_choice = st.selectbox(
        "Text Representation",
        ["TF-IDF Vectorizer", "Sentence Transformer Embeddings"],
        key="rep_choice"
    )
    emb_model = None

    if rep_choice == "Sentence Transformer Embeddings":
        emb_models = [
            "all-mpnet-base-v2",
            "all-MiniLM-L6-v2",
            "multi-qa-mpnet-base-dot-v1",
            "multi-qa-distilbert-dot-v1",
            "multi-qa-MiniLM-L6-dot-v1"
        ]
        emb_model = st.selectbox("Select Embedding Model", emb_models, key="emb_model")
    with st.form("cv_form"):
        classifiers = st.session_state.classifiers_templates
        selected_classifiers = st.multiselect(
            "Select Classifiers to Compare",
            list(classifiers.keys()),
            default=["Logistic Regression", "SVC"],
            key="selected_classifiers"
        )
        k = st.slider("Number of CV Folds", min_value=3, max_value=10, value=5, key="cv_k")

        submit_cv = st.form_submit_button("Run Cross-Validation")

    if submit_cv:
        results = []
        progress = st.progress(0)
        total_steps = max(1, len(selected_classifiers) * k)

        for idx_clf, clf_name in enumerate(selected_classifiers):
            clf_template = st.session_state.classifiers_templates[clf_name]
            fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'mcc': []}
            kf = KFold(n_splits=k, shuffle=True, random_state=42)

            for fold, (train_idx, val_idx) in enumerate(kf.split(symptoms)):
                X_train = [symptoms[i] for i in train_idx]
                y_train = [diseases[i] for i in train_idx]
                X_val = [symptoms[i] for i in val_idx]
                y_val = [diseases[i] for i in val_idx]

                if rep_choice == "TF-IDF Vectorizer":
                    vectorizer = TfidfVectorizer(max_features=5000)
                    X_train_vec = vectorizer.fit_transform(X_train)
                    X_val_vec = vectorizer.transform(X_val)
                else:
                    embedder = get_embedder(emb_model)
                    X_train_vec = embedder.encode(X_train, show_progress_bar=False)
                    X_val_vec = embedder.encode(X_val, show_progress_bar=False)

                clf = clf_template.__class__(**clf_template.get_params())
                clf.fit(X_train_vec, y_train)
                y_pred = clf.predict(X_val_vec)

                fold_metrics['accuracy'].append(accuracy_score(y_val, y_pred))
                fold_metrics['precision'].append(precision_score(y_val, y_pred, average='weighted', zero_division=0))
                fold_metrics['recall'].append(recall_score(y_val, y_pred, average='weighted', zero_division=0))
                fold_metrics['f1'].append(f1_score(y_val, y_pred, average='weighted', zero_division=0))
                fold_metrics['mcc'].append(matthews_corrcoef(y_val, y_pred))

                progress.progress(((idx_clf * k) + fold + 1) / total_steps)
            progress.empty()

            mean_metrics = {m: np.mean(v) for m, v in fold_metrics.items()}
            std_metrics = {m: np.std(v) for m, v in fold_metrics.items()}
            results.append({
                'Classifier': clf_name,
                'Accuracy Mean': f"{mean_metrics['accuracy']:.3f}",
                'Accuracy Std': f"{std_metrics['accuracy']:.3f}",
                'Precision Mean': f"{mean_metrics['precision']:.3f}",
                'Precision Std': f"{std_metrics['precision']:.3f}",
                'Recall Mean': f"{mean_metrics['recall']:.3f}",
                'Recall Std': f"{std_metrics['recall']:.3f}",
                'F1 Mean': f"{mean_metrics['f1']:.3f}",
                'F1 Std': f"{std_metrics['f1']:.3f}",
                'MCC Mean': f"{mean_metrics['mcc']:.3f}",
                'MCC Std': f"{std_metrics['mcc']:.3f}"
            })

        results_df = pd.DataFrame(results)
        st.session_state.cv_results_df = results_df
        st.session_state.cv_results_csv = results_df.to_csv(index=False).encode("utf-8")
        st.session_state.cv_complete = True

        if rep_choice == "TF-IDF Vectorizer":
            vectorizer = TfidfVectorizer(max_features=5000)
            X_vec = vectorizer.fit_transform(symptoms)
            st.session_state.vectorizer = vectorizer
            st.session_state.embedder = None
        else:
            embedder = get_embedder(emb_model)
            X_vec = embedder.encode(symptoms)
            st.session_state.embedder = embedder
            st.session_state.vectorizer = None

        st.session_state.X_vec = X_vec
        st.session_state.models = {}
        st.session_state.model_files = {}

        for clf_name in selected_classifiers:
            clf_template = st.session_state.classifiers_templates[clf_name]
            clf = clf_template.__class__(**clf_template.get_params())
            clf.fit(X_vec, diseases)
            st.session_state.models[clf_name] = clf

            files = {}
            b = io.BytesIO()
            pickle.dump(clf, b)
            files['pkl'] = b.getvalue()

            b = io.BytesIO()
            try:
                joblib.dump(clf, b)
                files['joblib'] = b.getvalue()
            except Exception:
                files['joblib'] = files['pkl']

            b = io.BytesIO()
            if hasattr(clf, 'coef_'):
                np.save(b, clf.coef_)
                files['npy'] = b.getvalue()
            elif hasattr(clf, 'feature_importances_'):
                np.save(b, clf.feature_importances_)
                files['npy'] = b.getvalue()
            else:
                files['npy'] = files['joblib']
            
            # Save which embedding model was used
            files['text_representation'] = rep_choice
            files['embedding_model'] = emb_model if rep_choice == "Sentence Transformer Embeddings" else None

            st.session_state.model_files[clf_name] = files

        st.success("Cross-validation finished. Models and download files are cached (client-side downloads available).")

    if st.session_state.cv_complete and st.session_state.cv_results_df is not None:
        st.subheader("Cross-Validation Results")
        st.dataframe(st.session_state.cv_results_df)

        csv_bytes = st.session_state.cv_results_csv
        st.markdown(get_download_link_bytes(csv_bytes, "cv_results.csv", mime="text/csv"), unsafe_allow_html=True)

        st.subheader("Download Trained Models (client-side links)")

        if 'model_files' in st.session_state and st.session_state.model_files:
            embedding_model_name = st.session_state.model_files.get('embedding_model', "TF-IDF")
            table_data = []
            for clf_name, files in st.session_state.model_files.items():
                row = {"Model": clf_name}
                if files.get('text_representation') == "Sentence Transformer Embeddings":
                    row["Embedding Model"] = files.get('embedding_model', "Unknown")
                else:
                    row["Embedding Model"] = "TF-IDF"
                row["PKL"] = get_download_link_bytes(files.get("pkl", b""), f"{clf_name}.pkl") if "pkl" in files else "N/A"
                row["Joblib"] = get_download_link_bytes(files.get("joblib", b""), f"{clf_name}.joblib") if "joblib" in files else "N/A"
                row["NPY"] = get_download_link_bytes(files.get("npy", b""), f"{clf_name}.npy") if "npy" in files else "N/A"
                table_data.append(row)

            table_df = pd.DataFrame(table_data)
            
            html_table = table_df.to_html(escape=False, index=False)
            st.markdown(html_table, unsafe_allow_html=True)
        else:
            st.info("Models not cached yet. Run cross-validation to generate downloadable files.")

        rep_choice_display = st.session_state.get('rep_choice', 'TF-IDF Vectorizer')
        emb_model_display = st.session_state.model_files.get('embedding_model', None)
        if emb_model_display and emb_model_display != "TF-IDF":
            st.info(f"Note: Downloads include only classifiers. Use SentenceTransformer('{emb_model_display}') to produce embeddings for new texts when reloading models.")
    else:
        st.info("👆 Run cross-validation first to see results and download models")