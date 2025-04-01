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

# --- CLASS LABELS & COLOUR MAPS ---
dataset_colormaps = {
    "Trento": {
        "Apples": "#FF0000",  # Red (0)
        "Buildings": "#00FF00",  # Green (1)
        "Ground": "#0000FF",  # Blue (2)
        "Woods": "#FFFF00",  # Yellow (3)
        "Vineyard": "#00FFFF",  # Cyan (4)
        "Roads": "#FF00FF"  # Magenta (5)
    },
    "MUUFL": {
        "Trees": "#FF0000",  # Red (0)
        "Grass_Pure": "#00FF00",  # Green (1)
        "Grass_Groundsurface": "#0000FF",  # Blue (2)
        "Dirt_And_Sand": "#FFFF00",  # Yellow (3)
        "Road_Materials": "#00FFFF",  # Cyan (4)
        "Water": "#FF00FF",  # Magenta (5)
        "Buildings'_Shadow": "#C0C0C0",  # Silver (6)
        "Buildings": "#808080",  # Gray (7)
        "Sidewalk": "#800000",  # Maroon (8)
        "Yellow_Curb": "#808000",  # Olive (9)
        "ClothPanels": "#008000"  # Dark Green (10)
    },
    "Houston18": {
        "Healthy grass": "#FF0000",  # Red (0)
        "Stressed grass": "#00FF00",  # Green (1)
        "Synthetic grass": "#0000FF",  # Blue (2)
        "Trees": "#FFFF00",  # Yellow (3)
        "Soil": "#00FFFF",  # Cyan (4)
        "Water": "#FF00FF",  # Magenta (5)
        "Residential": "#C0C0C0",  # Silver (6)
        "Commercial": "#808080",  # Gray (7)
        "Road": "#800000",  # Maroon (8)
        "Highway": "#808000",  # Olive (9)
        "Railway": "#008000",  # Dark Green (10)
        "Parking Lot 1": "#800080",  # Purple (11)
        "Parking Lot 2": "#008080",  # Teal (12)
        "Tennis Court": "#000080",  # Navy (13)
        "Running Track": "#FFA500"  # Orange (14)
    }
}


# Retrieve the correct colormap for the selected dataset
selected_colormap = dataset_colormaps.get(dataset_name, {})

# --- DISPLAY LEGEND ---
st.sidebar.header(f"{dataset_name} Class Legend")

legend_html = "<table style='border-collapse: collapse; width: 100%;'>"
for class_name, hex_color in selected_colormap.items():
    legend_html += f"""
    <tr>
        <td style='background-color: {hex_color}; width: 20px; height: 20px; border: 1px solid black;'></td>
        <td style='padding-left: 10px;'>{class_name}</td>
    </tr>
    """
legend_html += "</table>"

st.sidebar.markdown(legend_html, unsafe_allow_html=True)
