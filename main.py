# import streamlit as st
# import numpy as np
# from PIL import Image, ImageOps
# import io
# import pandas as pd
# from datetime import datetime
#
# # --- Model loader ---
# @st.cache_resource(show_spinner=False)
# def load_cnn_model(path: str = "skin_cancer_cnn.h5"):
#     try:
#         from tensorflow.keras.models import load_model
#         model = load_model(path)
#         return model
#     except Exception as e:
#         st.warning(f"Could not load model from {path}: {e}")
#         return None
#
# # --- Pillow compatibility ---
# def get_resample_method():
#     try:
#         return Image.Resampling.LANCZOS
#     except AttributeError:
#         return Image.LANCZOS
#
# # --- Preprocess ---
# def preprocess_image(pil_img: Image.Image, target_size=(224,224)) -> np.ndarray:
#     img = pil_img.convert('RGB')
#     resample = get_resample_method()
#     img = ImageOps.fit(img, target_size, method=resample)
#     arr = np.asarray(img).astype(np.float32) / 255.0
#     arr = np.expand_dims(arr, axis=0)
#     return arr
#
# # --- Prediction ---
# def predict_skin_cancer(pil_img: Image.Image, model):
#     if model is None:
#         return {"label": "Model not loaded", "score": None}
#     x = preprocess_image(pil_img)
#     pred = model.predict(x)
#     try:
#         score = float(np.asarray(pred).ravel()[0])
#     except Exception:
#         score = None
#     label = 'Unknown' if score is None else ('Malignant' if score > 0.5 else 'Benign')
#     return {"label": label, "score": score}
#
# # --- Cosmetic CSS ---
# NEON_CSS = """
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
# html, body, [class*="stApp"] { background: linear-gradient(180deg,#04050a 0%, #07102a 60%); color: #dfeefe; font-family: 'Inter', sans-serif; }
# .header { padding:12px; border-radius:12px; background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); box-shadow: 0 8px 30px rgba(0,0,0,0.6); }
# .h1 { font-size:26px; font-weight:700; background: linear-gradient(90deg,#6be3d1,#a88bff,#68b0ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
# .small { color:#9fb0d6; }
# .card { padding:10px; border-radius:10px; background: rgba(255,255,255,0.02); }
# .footer { color:#8fa4c0; font-size:12px; text-align:center }
# </style>
# """
#
# # --- Streamlit page setup ---
# st.set_page_config(page_title="SkinVision — Futuristic Detector", layout="wide", page_icon="🩺")
# st.markdown(NEON_CSS, unsafe_allow_html=True)
#
# # Sidebar
# with st.sidebar:
#     st.markdown("<div class='header'> <div class='h1'>SkinVision</div><div class='small'>Futuristic skin lesion classifier — demo only</div></div>", unsafe_allow_html=True)
#     st.markdown('---')
#     st.markdown('### Model')
#     model_path = st.text_input('Model path', value='skin_cancer_cnn.h5')
#     if st.button('Load model'):
#         st.session_state.model = load_cnn_model(model_path)
#         if st.session_state.model:
#             st.success('Model loaded')
#     st.markdown('---')
#     st.markdown('### Options')
#     demo_mode = st.checkbox('Demo mode (no embedded image)', value=False)
#     show_raw = st.checkbox('Show raw model score', value=False)
#     st.markdown('---')
#     st.markdown("<div class='small'>Not medical advice — demonstration only.</div>", unsafe_allow_html=True)
#
# # ensure session model
# if 'model' not in st.session_state:
#     st.session_state.model = load_cnn_model('skin_cancer_cnn.h5')
#
# # Layout
# col1, col2 = st.columns([1.3, 1])
# with col1:
#     st.markdown("<div class='header'><div class='h1'>Skin Cancer Detection</div><div class='small'>Upload a lesion photo or take one with your camera</div></div>", unsafe_allow_html=True)
#
#     uploaded = st.file_uploader('Choose an image (jpg/png)', type=['jpg','jpeg','png'])
#     camera_img = st.camera_input('Or take a photo (mobile)')
#
#     use_image = None
#     if camera_img is not None:
#         use_image = Image.open(camera_img).convert('RGB')
#     elif uploaded is not None:
#         use_image = Image.open(uploaded).convert('RGB')
#
#     if use_image is not None:
#         st.image(use_image, caption='Selected image', use_container_width=True)
#
#         if st.button('Analyze Image'):
#             with st.spinner('Running inference...'):
#                 result = predict_skin_cancer(use_image, st.session_state.model)
#
#             label = result.get('label')
#             score = result.get('score')
#
#             st.markdown("<div class='card'>", unsafe_allow_html=True)
#             st.subheader(f'Prediction: {label}')
#
#             if score is not None:
#                 # Create a clear probability breakdown for this image
#                 prob_malignant = float(np.clip(score, 0.0, 1.0))
#                 prob_benign = float(np.clip(1.0 - prob_malignant, 0.0, 1.0))
#
#                 # Show a bar chart built-in to Streamlit (no matplotlib dependency)
#                 chart_data = pd.DataFrame({
#                     'Class': ['Benign', 'Malignant'],
#                     'Probability': [prob_benign, prob_malignant]
#                 }).set_index('Class')
#
#                 st.write('**Probability breakdown for this image**')
#                 st.bar_chart(chart_data)
#
#                 # Display confidence and messages
#                 confidence = prob_malignant * 100.0 if label == 'Malignant' else prob_benign * 100.0
#                 st.write(f'Confidence: {confidence:.1f}%')
#                 if show_raw:
#                     st.write(f'Raw model output (prob malignant): {score:.4f}')
#
#                 if label == 'Malignant':
#                     st.error('Model indicates MALIGNANT — seek clinical evaluation.')
#                 else:
#                     st.success('Model indicates BENIGN — not a diagnosis.')
#
#             else:
#                 st.warning('No numeric score returned by model.')
#
#             st.markdown('</div>', unsafe_allow_html=True)
#
#             # Downloadable report
#             report = { 'timestamp': datetime.utcnow().isoformat() + 'Z', 'prediction': label, 'raw_score': score }
#             b = io.BytesIO()
#             b.write(str(report).encode('utf-8'))
#             b.seek(0)
#             st.download_button('Download report (txt)', data=b, file_name='skinvision_report.txt')
#
#     else:
#         if demo_mode:
#             st.info('Demo mode is ON but no embedded image is included. Please upload or capture an image to analyze.')
#         else:
#             st.info('Upload or capture an image to analyze.')
#
# with col2:
#     st.markdown("<div class='card'><h4>Quick Stats</h4>", unsafe_allow_html=True)
#     st.metric('Model loaded', 'Yes' if st.session_state.model is not None else 'No')
#     st.metric('Demo mode', 'On' if demo_mode else 'Off')
#     st.markdown('</div>', unsafe_allow_html=True)
#
#     st.markdown('### Tips')
#     st.write('- Use close-up, focused photos with uniform lighting.')
#     st.write('- Avoid shadows, heavy blur or excessive zoom.')
#     st.markdown('### Confidence Chart')
#     st.info('After analyzing an image, a per-image probability chart will appear showing Benign vs Malignant confidence.')
#
# st.markdown("<div class='footer'>Not medical advice — demonstration only.</div>", unsafe_allow_html=True)
#

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import io
import pandas as pd
from datetime import datetime

# --- Model loader ---
@st.cache_resource(show_spinner=False)
def load_cnn_model(path: str = "skin_cancer_cnn.h5"):
    try:
        from tensorflow.keras.models import load_model
        model = load_model(path)
        return model
    except Exception as e:
        st.warning(f"Could not load model from {path}: {e}")
        return None

# --- Pillow compatibility ---
def get_resample_method():
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS

# --- Preprocess ---
def preprocess_image(pil_img: Image.Image, target_size=(224,224)) -> np.ndarray:
    img = pil_img.convert('RGB')
    resample = get_resample_method()
    img = ImageOps.fit(img, target_size, method=resample)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# --- Prediction ---
def predict_skin_cancer(pil_img: Image.Image, model):
    if model is None:
        return {"label": "Model not loaded", "score": None}
    x = preprocess_image(pil_img)
    pred = model.predict(x)
    try:
        score = float(np.asarray(pred).ravel()[0])
    except Exception:
        score = None
    label = 'Unknown' if score is None else ('Malignant' if score > 0.5 else 'Benign')
    return {"label": label, "score": score}

# --- Cosmetic CSS ---
NEON_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="stApp"] { background: linear-gradient(180deg,#04050a 0%, #07102a 60%); color: #dfeefe; font-family: 'Inter', sans-serif; }
.header { padding:12px; border-radius:12px; background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); box-shadow: 0 8px 30px rgba(0,0,0,0.6); }
.h1 { font-size:26px; font-weight:700; background: linear-gradient(90deg,#6be3d1,#a88bff,#68b0ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.small { color:#9fb0d6; }
.card { padding:10px; border-radius:10px; background: rgba(255,255,255,0.02); }
.footer { color:#8fa4c0; font-size:12px; text-align:center }
</style>
"""

# --- Streamlit page setup ---
st.set_page_config(page_title="SkinVision — Futuristic Detector", layout="wide", page_icon="🩺")
st.markdown(NEON_CSS, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<div class='header'> <div class='h1'>SkinVision</div><div class='small'>Futuristic skin lesion classifier — demo only</div></div>", unsafe_allow_html=True)
    st.markdown('---')
    st.markdown('### Model')
    model_path = st.text_input('Model path', value='skin_cancer_cnn.h5')
    if st.button('Load model'):
        st.session_state.model = load_cnn_model(model_path)
        if st.session_state.model:
            st.success('Model loaded')
    st.markdown('---')
    st.markdown('### Options')
    demo_mode = st.checkbox('Demo mode (no embedded image)', value=False)
    show_raw = st.checkbox('Show raw model score', value=False)
    st.markdown('---')
    st.markdown("<div class='small'>Not medical advice — demonstration only.</div>", unsafe_allow_html=True)

# ensure session model
if 'model' not in st.session_state:
    st.session_state.model = load_cnn_model('skin_cancer_cnn.h5')

# Layout
col1, col2 = st.columns([1.3, 1])
with col1:
    st.markdown("<div class='header'><div class='h1'>Skin Cancer Detection</div><div class='small'>Upload a lesion photo or take one with your camera</div></div>", unsafe_allow_html=True)

    uploaded = st.file_uploader('Choose an image (jpg/png)', type=['jpg','jpeg','png'])
    camera_img = st.camera_input('Or take a photo (mobile)')

    use_image = None
    image_source = None  # NEW: track where the image came from

    if camera_img is not None:
        use_image = Image.open(camera_img).convert('RGB')
        image_source = "camera"
    elif uploaded is not None:
        use_image = Image.open(uploaded).convert('RGB')
        image_source = "upload"

    if use_image is not None:
        st.image(use_image, caption='Selected image', use_container_width=True)

        if st.button('Analyze Image'):
            # If image is from camera, do NOT run model — show message instead
            if image_source == "camera":
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Result")
                st.warning("Please provide a clear and correct image to analyze.")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                # Normal flow for uploaded images
                with st.spinner('Running inference...'):
                    result = predict_skin_cancer(use_image, st.session_state.model)

                label = result.get('label')
                score = result.get('score')

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader(f'Prediction: {label}')

                if score is not None:
                    # Create a clear probability breakdown for this image
                    prob_malignant = float(np.clip(score, 0.0, 1.0))
                    prob_benign = float(np.clip(1.0 - prob_malignant, 0.0, 1.0))

                    # Show a bar chart built-in to Streamlit (no matplotlib dependency)
                    chart_data = pd.DataFrame({
                        'Class': ['Benign', 'Malignant'],
                        'Probability': [prob_benign, prob_malignant]
                    }).set_index('Class')

                    st.write('**Probability breakdown for this image**')
                    st.bar_chart(chart_data)

                    # Display confidence and messages
                    confidence = prob_malignant * 100.0 if label == 'Malignant' else prob_benign * 100.0
                    st.write(f'Confidence: {confidence:.1f}%')
                    if show_raw:
                        st.write(f'Raw model output (prob malignant): {score:.4f}')

                    if label == 'Malignant':
                        st.error('Model indicates MALIGNANT — seek clinical evaluation.')
                    else:
                        st.success('Model indicates BENIGN — not a diagnosis.')

                else:
                    st.warning('No numeric score returned by model.')

                st.markdown('</div>', unsafe_allow_html=True)

                # Downloadable report
                report = {
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'prediction': label,
                    'raw_score': score
                }
                b = io.BytesIO()
                b.write(str(report).encode('utf-8'))
                b.seek(0)
                st.download_button('Download report (txt)', data=b, file_name='skinvision_report.txt')

    else:
        if demo_mode:
            st.info('Demo mode is ON but no embedded image is included. Please upload or capture an image to analyze.')
        else:
            st.info('Upload or capture an image to analyze.')

with col2:
    st.markdown("<div class='card'><h4>Quick Stats</h4>", unsafe_allow_html=True)
    st.metric('Model loaded', 'Yes' if st.session_state.model is not None else 'No')
    st.metric('Demo mode', 'On' if demo_mode else 'Off')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('### Tips')
    st.write('- Use close-up, focused photos with uniform lighting.')
    st.write('- Avoid shadows, heavy blur or excessive zoom.')
    st.markdown('### Confidence Chart')
    st.info('After analyzing an image, a per-image probability chart will appear showing Benign vs Malignant confidence (for uploaded images only).')

st.markdown("<div class='footer'>Not medical advice — demonstration only.</div>", unsafe_allow_html=True)

