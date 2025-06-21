import streamlit as st
import torch, torch.nn as nn
from torchvision.utils import make_grid
import numpy as np

# --- Model Architecture (must match training script) ---
latent_dim, num_classes = 100, 10
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(nn.Linear(latent_dim + num_classes, 256), nn.LeakyReLU(0.2, inplace=True), nn.Linear(256, 512), nn.LeakyReLU(0.2, inplace=True), nn.Linear(512, 1024), nn.LeakyReLU(0.2, inplace=True), nn.Linear(1024, 1 * 28 * 28), nn.Tanh())
    def forward(self, z, labels):
        img = self.model(torch.cat((self.label_embedding(labels), z), -1))
        return img.view(img.size(0), 1, 28, 28)

# --- Load Model ---
@st.cache_resource
def load_model():
    model = Generator()
    model.load_state_dict(torch.load('cgan_generator.pth', map_location=torch.device('cpu')))
    model.eval()
    return model
generator = load_model()

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Handwritten Digit Generation Web App")
st.write("Select a digit and click 'Generate' to create 5 new images.")
col1, col2 = st.columns([1, 4])
with col1:
    digit_to_generate = st.selectbox("Select a digit (0-9):", options=list(range(10)))
    generate_button = st.button("Generate Images", type="primary")

# --- Generate and Display ---
if generate_button:
    with torch.no_grad():
        z = torch.randn(5, latent_dim)
        labels = torch.LongTensor([digit_to_generate] * 5)
        generated_images = generator(z, labels) * 0.5 + 0.5
        grid = make_grid(generated_images, nrow=5, padding=10, pad_value=1)
        img_grid_np = grid.permute(1, 2, 0).numpy()
        with col2:
            st.subheader(f"Generated Images of Digit: {digit_to_generate}")
            st.image(img_grid_np, width=600)
else:
     with col2:
        st.info("Select a digit to generate.")