import streamlit as st

def home_page():
    """
    Render the Home page of the Streamlit app.
    """
    st.header("🏠 Home Page")
    st.write("Welcome! This is a demo biomedical ML application.")
    st.subheader("Description")
    st.markdown("""
    - This Streamlit app demonstrates ML pipelines in biomedicine.  
    - It includes multiple modules: EDA, genomics, imaging, or NLP.  
    - Use the tabs above to switch between sections.  
    """)
    st.markdown("""
    Watch a quick demo of the app's features, including genomic analysis and NLP for disease prediction.
    """)
    # Embed local video
    try:
        video_file = open("app/demo/Demo.mov", "rb")
        video_bytes = video_file.read()
        st.video(video_bytes)
    except FileNotFoundError:
        st.error("Demo video not found at 'app/demo/Demo.mov'. Please ensure the file exists.")



