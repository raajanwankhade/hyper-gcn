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
        "Apples": (255, 0, 0),  # Red (0)
        "Buildings": (0, 255, 0),  # Green (1)
        "Ground": (0, 0, 255),  # Blue (2)
        "Woods": (255, 255, 0),  # Yellow (3)
        "Vineyard": (0, 255, 255),  # Cyan (4)
        "Roads": (255, 0, 255)  # Magenta (5)
    },
    "MUUFL": {
        "Trees": (255, 0, 0),  # Red (0)
        "Grass_Pure": (0, 255, 0),  # Green (1)
        "Grass_Groundsurface": (0, 0, 255),  # Blue (2)
        "Dirt_And_Sand": (255, 255, 0),  # Yellow (3)
        "Road_Materials": (0, 255, 255),  # Cyan (4)
        "Water": (255, 0, 255),  # Magenta (5)
        "Buildings'_Shadow": (192, 192, 192),  # Silver (6)
        "Buildings": (128, 128, 128),  # Gray (7)
        "Sidewalk": (128, 0, 0),  # Maroon (8)
        "Yellow_Curb": (128, 128, 0),  # Olive (9)
        "ClothPanels": (0, 128, 0)  # Dark Green (10)
    },
    "Houston18": {
        "Healthy grass": (255, 0, 0),  # Red (0)
        "Stressed grass": (0, 255, 0),  # Green (1)
        "Synthetic grass": (0, 0, 255),  # Blue (2)
        "Trees": (255, 255, 0),  # Yellow (3)
        "Soil": (0, 255, 255),  # Cyan (4)
        "Water": (255, 0, 255),  # Magenta (5)
        "Residential": (192, 192, 192),  # Silver (6)
        "Commercial": (128, 128, 128),  # Gray (7)
        "Road": (128, 0, 0),  # Maroon (8)
        "Highway": (128, 128, 0),  # Olive (9)
        "Railway": (0, 128, 0),  # Dark Green (10)
        "Parking Lot 1": (128, 0, 128),  # Purple (11)
        "Parking Lot 2": (0, 128, 128),  # Teal (12)
        "Tennis Court": (0, 0, 128),  # Navy (13)
        "Running Track": (255, 165, 0)  # Orange (14)
    }
}


def generate_legend_html(dataset_name, dataset_colormap):
    legend_html = """
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid black;
            padding: 5px;
            text-align: left;
        }
        .color-box {
            width: 20px;
            height: 20px;
            display: inline-block;
            border: 1px solid black;
        }
    </style>
    <h3>Legend - {} Dataset</h3>
    <table>
        <tr>
            <th>Class</th>
            <th>Colour</th>
        </tr>
    """.format(dataset_name)
    
    for class_name, rgb in dataset_colormap.items():
        color_style = "background-color: rgb({}, {}, {});".format(*rgb)
        legend_html += """
        <tr>
            <td>{}</td>
            <td><span class='color-box' style='{}'></span></td>
        </tr>
        """.format(class_name, color_style)

    legend_html += "</table>"
    return legend_html

st.sidebar.markdown(generate_legend_html(dataset_name, dataset_colormaps[dataset_name]), unsafe_allow_html=True)
