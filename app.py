import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from src.main import HybridUNet

# Page Configuration with clean light styling layout
st.set_page_config(
    page_title="Hybrid Quantum-Classical Road Extraction",
    page_icon="🛰️",
    layout="wide"
)

# Custom Light Theme Styling Injection
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        color: #212529;
    }
    .stSidebar {
        background-color: #e9ecef;
    }
    h1, h2, h3 {
        color: #1d3557;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.title("🛰️ Hybrid Quantum-Classical Urban Mobility Analysis")
st.markdown("**VTU-Qiskit Fall Fest 2026 | Team zen | Track 4: Quantum + AI & ML**")
st.markdown("---")

# Sidebar Information
st.sidebar.header("Architecture Overview")
st.sidebar.info(
    "**Backend:** PyTorch Dual-Attention U-Net\n\n"
    "**Quantum Bottleneck:** Qiskit Variational Quantum Circuit (VQC)\n\n"
    "**Objective:** High-resolution satellite imagery road segmentation with reduced parameter dimensionality."
)

# Load the Trained Quantum Model
@st.cache_resource
def load_quantum_model():
    model = HybridUNet()
    try:
        model.load_state_dict(torch.load("hybrid_unet_model.pth", map_location=torch.device('cpu'), weights_only=True))
        model.eval()
        return model, True
    except Exception:
        try:
            model.load_state_dict(torch.load("hybrid_unet_model.pth", map_location=torch.device('cpu')))
            model.eval()
            return model, True
        except FileNotFoundError:
            return None, False

model, model_loaded = load_quantum_model()

if not model_loaded:
    st.warning("⚠️ Warning: 'hybrid_unet_model.pth' not found in root directory. Please run `python -m src.optimizer` first to generate model weights.")

# Main Interface: Image Upload & Processing
uploaded_file = st.file_uploader("Upload a High-Resolution Satellite Image (.jpg, .png)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    # Display Original Input
    raw_image = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.subheader("📍 Raw Satellite Imagery")
        st.image(raw_image, use_container_width=True)
        
    # Process through Hybrid Quantum-Classical Pipeline
    with col2:
        st.subheader("⚡ Quantum Extracted Road Network")
        if model_loaded:
            with st.spinner("Executing Quantum Tensor Transformation & Inference..."):
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor()
                ])
                input_tensor = transform(raw_image).unsqueeze(0)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    prediction_mask = (output > 0.5).float().squeeze().numpy()
                
                st.image(prediction_mask, use_container_width=True, clamp=True)
                st.success("Quantum spatial extraction completed successfully!")
        else:
            st.error("Inference skipped: Model weights missing.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'>Team zen · Built with ⚛️ Qiskit & PyTorch</div>", unsafe_allow_html=True)
