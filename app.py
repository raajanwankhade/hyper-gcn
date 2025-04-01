import streamlit as st
import os
import base64
from PIL import Image
import subprocess
import pandas as pd

def set_background(image_path):
    """Set a background image for the Streamlit app."""
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{image_path}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

def get_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

dataset_colormaps = {
    "Trento": {
        "Apples": (255, 0, 0),
        "Buildings": (0, 255, 0),
        "Ground": (0, 0, 255),
        "Woods": (255, 255, 0),
        "Vineyard": (0, 255, 255),
        "Roads": (255, 0, 255)
    },
    "MUUFL": {
        "Trees": (255, 0, 0),
        "Grass_Pure": (0, 255, 0),
        "Grass_Groundsurface": (0, 0, 255),
        "Dirt_And_Sand": (255, 255, 0),
        "Road_Materials": (0, 255, 255),
        "Water": (255, 0, 255),
        "Buildings'_Shadow": (192, 192, 192),
        "Buildings": (128, 128, 128),
        "Sidewalk": (128, 0, 0),
        "Yellow_Curb": (128, 128, 0),
        "ClothPanels": (0, 128, 0)
    },
    "Houston18": {
        "Healthy grass": (255, 0, 0),
        "Stressed grass": (0, 255, 0),
        "Synthetic grass": (0, 0, 255),
        "Trees": (255, 255, 0),
        "Soil": (0, 255, 255),
        "Water": (255, 0, 255),
        "Residential": (192, 192, 192),
        "Commercial": (128, 128, 128),
        "Road": (128, 0, 0),
        "Highway": (128, 128, 0),
        "Railway": (0, 128, 0),
        "Parking Lot 1": (128, 0, 128),
        "Parking Lot 2": (0, 128, 128),
        "Tennis Court": (0, 0, 128),
        "Running Track": (255, 165, 0)
    }
}


background_image_path = r"bigmap.jpg"
set_background(get_image_as_base64(background_image_path))

st.title("National Institute of Technology Karnataka, Surathkal")
st.subheader("Department of Information Technology")
st.subheader("IT353: Deep Learning, B. Tech. in Artificial Intelligence")

st.header("Hyperspectral Image Classification with Attention Graph Convolutional Network")
st.subheader("Team Members: Bhuvanesh Singla (221AI014), Raajan Rajesh Wankhade (221AI031)")

# dropdown for dataset selection
dataset_name = st.selectbox("Select Dataset", ["MUUFL", "Houston18", "Trento" ])
processing_type = st.selectbox("Select Processing Type", ["Show pre-computed result", "Run through model (20-30 Minutes)"])

model_path = f"models/{dataset_name}_weights.pt" 

rgb_image_path = os.path.join(r"rgbs", f"{dataset_name}_rgb.png")
if os.path.exists(rgb_image_path):
    st.image(rgb_image_path, caption=f"{dataset_name} RGB Image", use_container_width =True)
else:
    st.warning(f"RGB image for {dataset_name} not found.")

if st.button("Run/Show Results"):
    save_path = rf"results/{dataset_name}_result.png"
    if processing_type == "Show pre-computed result":
        if os.path.exists(save_path):
            st.image(save_path, caption=f"Result for {dataset_name}", use_container_width =True)
        else:
            st.error("Result not found!")
    else:
        st.write("Running model... This may take some time.")
        subprocess.run(["python", "infer_script.py", dataset_name, model_path])
        
        output_image_path = os.path.join("live_results", f"{dataset_name}_live_results.png")
    
        if os.path.exists(output_image_path):
            st.image(output_image_path, caption="Predicted Classification Map", use_container_width =True)
        else:
            st.write("Result image not found. Try running the model first.")