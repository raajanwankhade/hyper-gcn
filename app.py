import streamlit as st
import os
from PIL import Image

st.title("National Institute of Technology Karnataka, Surathkal")
st.subheader("Department of Information Technology")
st.subheader("IT353: Deep Learning, B. Tech. in Artificial Intelligence")

st.header("Hyperspectral Image Classification with Attention Graph Convolutional Network")
st.subheader("Team Members: Bhuvanesh Singla (221AI014), Raajan Rajesh Wankhade (221AI031)")

# dropdown for dataset selection
dataset_name = st.selectbox("Select Dataset", ["MUUFL","Trento", "Houston"])
processing_type = st.selectbox("Select Processing Type", ["Show pre-computed result", "Run through model (20-30 Minutes)"])

if st.button("Run/Show Results"):
    save_path = rf"D:\All NITK\Semester 6\IT353 - Deep Learning\Project\hyperspectral_classification_application\results\{dataset_name}_result.png"
    if processing_type == "Show pre-computed result":
        if os.path.exists(save_path):
            st.image(save_path, caption=f"Result for {dataset_name}")
        else:
            st.error("Result not found!")
    # else:
    #     model = load_model()  # Implement model loading
    #     output_path = predict_and_save_grid(dataset_name, model, save_path)
    #     st.image(output_path, caption=f"Model-generated result for {dataset_name}")
