import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim

st.set_page_config(page_title="AI Robustness Audit", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_audit_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(DEVICE)
    model.eval()
    
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    return model, feature_extractor


class AttackPipeline:
    @staticmethod
    def gaussian_noise(img_tensor, intensity):
        noise = torch.randn_like(img_tensor) * intensity
        return torch.clamp(img_tensor + noise, 0, 1)

    @staticmethod
    def patch_attack(img_tensor, size_pct):
        adv_img = img_tensor.clone()
        _, _, h, w = adv_img.shape
        size = int(min(h, w) * size_pct)
        # Strategic placement: center-weighted
        y, x = (h-size)//2, (w-size)//2
        adv_img[:, :, y:y+size, x:x+size] = torch.rand(3, size, size)
        return adv_img

# --- 3. MATHEMATICAL METRICS ---
def compute_robustness_metrics(orig_t, adv_t, feat_model):
    # SSIM for structural change
    orig_np = orig_t.squeeze().permute(1, 2, 0).cpu().numpy()
    adv_np = adv_t.squeeze().permute(1, 2, 0).cpu().numpy()
    score_ssim, _ = ssim(orig_np, adv_np, full=True, channel_axis=2, data_range=1.0)
    
    # Feature Embedding Distance (Cosine Similarity)
    with torch.no_grad():
        feat_orig = feat_model(orig_t).flatten(1)
        feat_adv = feat_model(adv_t).flatten(1)
        cos_sim = torch.nn.functional.cosine_similarity(feat_orig, feat_adv).item()
    
    return score_ssim, cos_sim

# --- 4. DASHBOARD UI ---
st.title("🛡️ Enterprise AI Robustness Audit")
model, extractor = load_audit_model()

uploaded_file = st.sidebar.file_uploader("Upload Target Image", type=["jpg", "png"])
attack_choice = st.sidebar.selectbox("Attack Strategy", ["Gaussian Noise", "Adversarial Patch"])
intensity = st.sidebar.slider("Perturbation Intensity", 0.0, 0.5, 0.1)

if uploaded_file:
    # Pre-processing
    img = Image.open(uploaded_file).convert('RGB')
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    img_t = transform(img).unsqueeze(0).to(DEVICE)

    # Execute Audit
    if attack_choice == "Gaussian Noise":
        adv_t = AttackPipeline.gaussian_noise(img_t, intensity)
    else:
        adv_t = AttackPipeline.patch_attack(img_t, intensity)

    # Metrics Computation
    ssim_val, cos_val = compute_robustness_metrics(img_t, adv_t, extractor)
    
    # Visualization & Results
    col1, col2, col3 = st.columns(3)
    col1.image(img, caption="Original Input", use_container_width=True)
    col2.image(transforms.ToPILImage()(adv_t.squeeze()), caption="Perturbed Input", use_container_width=True)
    
    # Difference Map (Visual identification)
    diff = torch.abs(img_t - adv_t).squeeze().permute(1, 2, 0).cpu().numpy()
    col3.image(diff / diff.max(), caption="Perturbation Map", use_container_width=True)

    # --- 5. ROBUSTNESS SCORING ---
    st.header("📊 Security Audit Report")
    m1, m2, m3 = st.columns(3)
    
    # Normalized Robustness Score: Harmonic mean of SSIM and Cosine Similarity
    robustness_score = (2 * ssim_val * cos_val) / (ssim_val + cos_val)
    
    m1.metric("Structural Integrity (SSIM)", f"{ssim_val:.3f}")
    m2.metric("Latent Similarity (Cosine)", f"{cos_val:.3f}")
    m3.metric("Final Robustness Score", f"{robustness_score:.2%}")

    if robustness_score < 0.85:
        st.error("🚨 HIGH VULNERABILITY: Model embeddings shifted significantly with minimal input change.")
    else:
        st.success("✅ RESILIENT: Model maintained latent consistency despite perturbations.")