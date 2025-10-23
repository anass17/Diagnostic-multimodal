import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F


st.set_page_config(page_title="Classification Cellules Cancéreuses", page_icon="🧬", layout="centered")

st.title("Classification des cellules Sanguines Cancéreuses")
st.write("Uploader une image de cellule pour prédiction avec le modèle GoogleNet entraîné.")


# Charger le modèle : 
@st.cache_resource
def load_lodel(model_path, num_classes):
    model = models.googlenet(pretrained=False, aux_logits=False)
    # model = models.googlenet(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
        nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


MODEL_PATH = "./models/blood_cells_googlenet_model.pth"
CLASSES = ['Benign', 'Pre-B', 'Pro-B', 'early Pre-B']

try:
    model = load_lodel(MODEL_PATH, len(CLASSES))
    st.success("Modèle chargé avec succès !")
    # st.write(model.fc)
except Exception as e :
    st.error(f"Erreur lors du chargement du modèle : {e}")
    
# définir Le Transformation : 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# Charger l'image

uploded_file = st.file_uploader("Choisir une image de cellule", type=["jpg", "jpeg", "png"])

if uploded_file is not None:
    image = Image.open(uploded_file).convert("RGB")
    st.image(image, caption="Image Chargée", use_container_width=True)
    
    input_image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_image)
        probabilities = torch.exp(output)
        confidence, predicted_class = torch.max(probabilities, 1)
        
    st.subheader("Résultat de la Prédiction : ")
    st.write(f"**Classe prédite :** {CLASSES[predicted_class.item()]}")
    st.write(f"**Confiance :** {confidence.item() * 100:.2f}%")
    
    st.bar_chart({CLASSES[i]: probabilities[0][i].item() for i in range(len(CLASSES))})