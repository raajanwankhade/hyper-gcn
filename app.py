import streamlit as st
import os
import base64
from PIL import Image
import subprocess

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

background_image_path = r"airbus_Sentinel-2.jpg"
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
