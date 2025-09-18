#!/bin/bash
# Convenience script to run the ML for Biomedicine Streamlit app

# Activate Conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate streamlit_app_mlbm

# Check if environment is activated
if [ $? -ne 0 ]; then
    echo "Failed to activate Conda environment 'streamlit_app_mlbm'. Please ensure it exists."
    exit 1
fi

# Run the Streamlit app
streamlit run app/app.py