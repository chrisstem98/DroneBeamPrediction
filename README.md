# UAV Beam Prediction and Handover Detection

## Folder Structure Required

    uav_beam_project/
        scenario23_dev/          <- extracted zip from DeepSense 6G
            scenario23.csv
            unit1/
            unit2/
            resources/
        step1_load_data.py
        step2_preprocessing.py
        step3_ml_models.py
        step4_gemini.py
        step5_handover.py
        step6_final_comparison.py
        requirements.txt

## Setup

    pip install -r requirements.txt

## Important
    In step1_load_data.py set ROOT to the name of your extracted folder.
    Default is ROOT = 'scenario23_dev'

    In step4_gemini.py insert your Gemini API key:
    GEMINI_API_KEY = "YOUR_KEY_HERE"
    Get a free key from: https://aistudio.google.com

## Execution Order

    python step1_load_data.py
    python step2_preprocessing.py
    python step3_ml_models.py
    python step4_gemini.py
    python step5_handover.py
    python step6_final_comparison.py

## Reference
Lu Cheng, Hongliang Zhang, Boya Di, Dusit Niyato, Lingyang Song,
"Large Language Models Empower Multimodal Integrated Sensing
and Communication," IEEE Communications Magazine, May 2025.