import streamlit as st
import os
import base64
# from PIL import Image
import torch
# import subprocess
import infer_script 
from infer_script import AttentionGCN, CoAtNetRelativeAttention
import pickle
st.set_option('server.runOnSave', False)
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


background_image_path = r"bigmap.jpg"
set_background(get_image_as_base64(background_image_path))


st.title("National Institute of Technology Karnataka, Surathkal")
st.subheader("Department of Information Technology")
st.subheader("IT353: Deep Learning, B. Tech. in Artificial Intelligence")

st.header("Hyperspectral Image Classification with Attention Graph Convolutional Network")
st.subheader("Team Members: Bhuvanesh Singla (221AI014), Raajan Rajesh Wankhade (221AI031)")

# dropdown for dataset selection
dataset_name = st.selectbox("Select Dataset", ["MUUFL", "Trento" ])
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
        # subprocess.run(["python", "infer_script.py", dataset_name, model_path])
        X, y = infer_script.loadData(dataset_name)
        
        if dataset_name == "MUUFL":
            CLASSES_NUM = 11
        else:
            CLASSES_NUM = 6 # Trento
        
        pre_height = 8
        pre_width = 8
        in_dim = 8
        proj_dim = 8
        head_dim = 4
        n_classes = CLASSES_NUM
        attention_dropout = 0.1
        ff_dropout = 0.1    
        
        torch.serialization.add_safe_globals([AttentionGCN])
        torch.serialization.add_safe_globals([CoAtNetRelativeAttention])
        
        model = AttentionGCN(pre_height, pre_width, in_dim, proj_dim, head_dim, n_classes, attention_dropout, ff_dropout)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        all_data = infer_script.PatchSet(X, y, infer_script.PATCH_SIZE,is_pred = True)
        all_loader = infer_script.DataLoader(all_data,infer_script.BATCH_SIZE,shuffle= False)
        
        model = torch.load(model_path,pickle_module=pickle, map_location = device, weights_only=False)
        
        infer_script.predict_and_save_grid(dataset_name, all_data, model, "prediction_map.png")
        st.write("Model has finished running.")
        output_image_path = os.path.join("live_results", f"{dataset_name}_live_results.png")
    
        if os.path.exists(output_image_path):
            st.image(output_image_path, caption="Predicted Classification Map", use_container_width =True)
        else:
            st.write("Result image not found. Try running the model first.")