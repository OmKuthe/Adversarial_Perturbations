# demo_streamlit.py
# Streamlit interactive demo for "snyk-adversarial-inputs-to-image-classifiers"
# Provides quick FGSM + simple transforms, shows predictions, perturbation and saliency map.

import io
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import streamlit as st
import matplotlib.pyplot as plt

# ---------- Utilities ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Adversarial Demo", layout="wide")

@st.cache_resource
def load_model():
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
        model.eval()
        model.to(DEVICE)
        categories = weights.meta.get("categories", None)  # list of ImageNet labels (if available)
    except Exception as e:
        # fallback: load without weights (rare) - still works but predictions meaningless
        st.warning("Could not load weights automatically. Ensure internet or local cache exists.")
        model = models.resnet50()
        model.eval()
        model.to(DEVICE)
        categories = None
    return model, categories

model, IMAGENET_LABELS = load_model()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def pil_to_tensor(img_pil):
    return preprocess(img_pil).unsqueeze(0).to(DEVICE)

def predict_topk(img_tensor, k=3):
    with torch.no_grad():
        out = model(img_tensor)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
    topk_idx = probs.argsort()[-k:][::-1]
    results = []
    for idx in topk_idx:
        label = str(idx)
        if IMAGENET_LABELS:
            try:
                label = IMAGENET_LABELS[idx]
            except Exception:
                label = str(idx)
        results.append((label, float(probs[idx])))
    return results

def tensor_to_pil(tensor):
    # tensor single-batch normalized -> convert to PIL
    t = tensor.squeeze(0).cpu()
    # unnormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img = t * std + mean
    img = img.clamp(0,1)
    arr = (img.numpy().transpose(1,2,0)*255).astype(np.uint8)
    return Image.fromarray(arr)

# ---------- Attacks & Transforms ----------
def fgsm_attack(image_tensor, epsilon=0.01):
    # image_tensor: requires grad? shape 1x3x224x224
    image = image_tensor.clone().detach().to(DEVICE)
    image.requires_grad = True
    out = model(image)
    pred = out.argmax(dim=1)
    loss = F.nll_loss(F.log_softmax(out, dim=1), pred)
    model.zero_grad()
    loss.backward()
    sign_grad = image.grad.sign()
    adv = image + epsilon * sign_grad
    # clamp in normalized space: convert to unnormalized bounds between 0 and 1
    # easiest: convert back to PIL & re-preprocess to keep pipeline consistent
    adv = torch.clamp(adv, -3.0, 3.0)  # safe clamp in normalized units
    return adv.detach()

def apply_blur(pil_img, radius=3):
    return pil_img.filter(ImageFilter.GaussianBlur(radius))

def apply_brightness(pil_img, delta=1.2):
    enhancer = ImageEnhance.Brightness(pil_img)
    return enhancer.enhance(delta)

def apply_rotation(pil_img, angle=15):
    return pil_img.rotate(angle, resample=Image.BILINEAR)

def apply_jpeg_compress(pil_img, quality=50):
    bio = io.BytesIO()
    pil_img.save(bio, format="JPEG", quality=int(quality))
    bio.seek(0)
    return Image.open(bio).convert("RGB")

# Saliency map: gradient of top prediction w.r.t. input (simple)
def compute_saliency(img_tensor):
    img = img_tensor.clone().detach().to(DEVICE)
    img.requires_grad = True
    scores = model(img)
    top_idx = scores.argmax(dim=1)
    score = scores[0, top_idx]
    model.zero_grad()
    score.backward()
    sal = img.grad.abs().detach().cpu().squeeze().numpy()
    sal = sal.max(axis=0)
    # normalize sal to 0-255
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    sal_img = (sal * 255).astype(np.uint8)
    sal_pil = Image.fromarray(sal_img).resize((224,224)).convert("L")
    return sal_pil

# ---------- Streamlit UI ----------
st.title("Interactive Adversarial / Input Variation Demo — ResNet50")
st.write("Upload an image or use repo's sample. Choose attack/transform and tweak severity. "
         "Shows predictions, confidence, perturbation and saliency.")

col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Input")
    uploaded = st.file_uploader("Upload image (PNG/JPG)", type=["png","jpg","jpeg"])
    use_repo_sample = st.checkbox("Use repo sample 'results_real_image/original.png' if exists", value=True)
    sample_path = Path("results_real_image/original.png")
    if uploaded:
        pil = Image.open(uploaded).convert("RGB")
    elif use_repo_sample and sample_path.exists():
        pil = Image.open(sample_path).convert("RGB")
    else:
        # fallback sample: create a simple RGB square
        pil = Image.new("RGB", (224,224), (120,120,200))
        st.info("No image uploaded and repo sample not found. Using placeholder.")

    st.image(pil, caption="Original image", use_container_width=True)

    st.subheader("Choose attack / transform")
    attack = st.selectbox("Attack / Transform", ["None", "FGSM (fast adversarial)", "Blur", "Brightness", "Rotation", "JPEG Compression"])
    if attack == "FGSM":
        eps = st.slider("FGSM: epsilon (strength)", 0.0, 0.2, 0.01, 0.005)
    elif attack == "Blur":
        radius = st.slider("Blur radius", 1, 15, 3)
    elif attack == "Brightness":
        delta = st.slider("Brightness multiplier", 0.2, 2.0, 1.2, 0.1)
    elif attack == "Rotation":
        angle = st.slider("Rotation angle (degrees)", -90, 90, 15)
    elif attack == "JPEG Compression":
        quality = st.slider("JPEG quality (lower = stronger)", 5, 100, 50)

    run_button = st.button("Run attack / transform")

with col2:
    st.subheader("Results")
    # place holders
    pred_placeholder = st.empty()
    imgs_cols = st.columns(3)
    orig_col, pert_col, diff_col = imgs_cols

    sal_col = st.empty()
    table_col = st.empty()

if run_button:
    # preprocess and predict original
    img_t = pil_to_tensor(pil)
    orig_preds = predict_topk(img_t, k=3)
    pred_lines = ["### Original predictions"]
    for label, prob in orig_preds:
        pred_lines.append(f"- `{label}` — {prob*100:.2f}%")
    pred_placeholder.markdown("\n".join(pred_lines))

    # apply chosen transform
    if attack == "None":
        pert_pil = pil
    elif attack == "FGSM":
        adv_t = fgsm_attack(img_t, epsilon=eps)
        pert_pil = tensor_to_pil(adv_t)
    elif attack == "Blur":
        pert_pil = apply_blur(pil, radius=radius)
    elif attack == "Brightness":
        pert_pil = apply_brightness(pil, delta=delta)
    elif attack == "Rotation":
        pert_pil = apply_rotation(pil, angle=angle)
    elif attack == "JPEG Compression":
        pert_pil = apply_jpeg_compress(pil, quality=quality)
    else:
        pert_pil = pil

    # make sure pert_pil is RGB PIL
    pert_pil = pert_pil.convert("RGB")

    # compute perturbed tensor and predictions
    pert_t = pil_to_tensor(pert_pil)
    pert_preds = predict_topk(pert_t, k=3)

    # save images in-memory and show side-by-side
    orig_col.image(pil.resize((300,300)), caption="Original", use_container_width=False)
    pert_col.image(pert_pil.resize((300,300)), caption=f"Perturbed ({attack})", use_container_width=False)

    # difference image (visualize perturbation)
    # convert to numpy arrays (0..255) same size
    o = np.array(pil.resize((224,224))).astype(np.int16)
    p = np.array(pert_pil.resize((224,224))).astype(np.int16)
    diff = np.clip(np.abs(o - p).astype(np.uint8)*4, 0,255)  # amplify diff for visibility
    diff_pil = Image.fromarray(diff).resize((300,300))
    diff_col.image(diff_pil, caption="Perturbation ×4 (abs diff)", use_container_width=False)

    # saliency map (gradient-based) using perturbed input (shows what model focused on)
    try:
        sal = compute_saliency(pert_t)
        sal_col.image(sal.resize((300,300)), caption="Saliency map (gradient of top pred)", use_container_width=False)
    except Exception as e:
        sal_col.write("Saliency map generation failed: " + str(e))

    # show perturbed predictions in table
    lines = ["### Perturbed predictions"]
    for label, prob in pert_preds:
        lines.append(f"- `{label}` — {prob*100:.2f}%")
    table_col.markdown("\n".join(lines))

    # compact result table comparing top-1
    orig_top = orig_preds[0][0], orig_preds[0][1]
    pert_top = pert_preds[0][0], pert_preds[0][1]

    st.markdown("### Summary (Top-1 comparison)")
    st.table({
        "Item": ["Original top-1", "Perturbed top-1"],
        "Label": [orig_top[0], pert_top[0]],
        "Confidence (%)": [f"{orig_top[1]*100:.2f}", f"{pert_top[1]*100:.2f}"]
    })

st.markdown("---")
st.write("Notes: FGSM here is single-step (fast) — good for demos. DeepFool / PGD are iterative and slow on CPU. "
         "Saliency map is a simple gradient-based map (useful to see what pixels influenced the top prediction).")
