import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import get_model

CLASSES = ['diseased', 'healthy']  # alphabetical order!

@st.cache_resource
def load_model():
    model = get_model(num_classes=2)
    model.load_state_dict(
        torch.load('crop_model.pth', map_location='cpu'))
    model.eval()
    return model

def predict(image, model):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]),
    ])
    tensor = tf(image).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
    probs = torch.softmax(out, dim=1)[0]
    idx   = probs.argmax().item()
    return CLASSES[idx], float(probs[idx])

st.title("🌿 Crop Disease Detector")
st.write("Upload a leaf photo to check if the plant is healthy.")

model = load_model()
file  = st.file_uploader("Upload leaf image",
                          type=['jpg','jpeg','png'])
if file:
    img  = Image.open(file).convert('RGB')
    st.image(img, caption="Your leaf", use_column_width=True)
    label, conf = predict(img, model)
    if label == 'healthy':
        st.success(f"✅ Healthy — {conf*100:.1f}% confident")
    else:
        st.error(f"⚠️ Diseased — {conf*100:.1f}% confident")