import streamlit as st
from streamlit_cropper import st_cropper
import cv2
from PIL import Image
import os

st.set_page_config(
     page_title='Webtoon Image Crop App',
)

# pip install streamlit_cropper

st.title("Webtoon Image Crop App ✨")
st.markdown("---")
st.markdown("")

current_path = '.'


# select webtoon
import natsort
webtoon = st.sidebar.selectbox('webtoon',
                    natsort.natsorted(os.listdir(current_path)))

# success image
webtoon_align_path = f'{current_path}/{webtoon}/align-success'
image_align = st.sidebar.selectbox('align success image',
                    os.listdir(webtoon_align_path)
                    )

# fail webtoon
webtoon_path = f'{current_path}/{webtoon}/align-fail'
fail_webtoon = st.sidebar.radio('fail webtoon',
                    os.listdir(webtoon_path))

# image ratio
aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
aspect_dict = {"1:1": (1,1),
                "16:9": (16,9),
                "4:3": (4,3),
                "2:3": (2,3),
                "Free": None}
aspect_ratio = aspect_dict[aspect_choice]

realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
if not realtime_update:
    st.write("Double click to save crop")

degree = st.sidebar.slider('rotate', -45, 45, 0, 1)

# ===========================


img_file = f'{webtoon_path}/{fail_webtoon}'
image = cv2.imread(img_file)

(h, w) = image.shape[:2]
(cX, cY) = (w / 2, h / 2)
 
M = cv2.getRotationMatrix2D((cX, cY), degree, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))

color_coverted = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
pil_image=Image.fromarray(color_coverted)


col1, col2 = st.columns(2)
with col1:
    cropped_img = st_cropper(pil_image, realtime_update=True,
                        aspect_ratio=aspect_ratio)

with col2:
    st.image(f'{webtoon_align_path}/{image_align}', width=256)
    
st.markdown("---")
_ = cropped_img.thumbnail((256,256))

col1, col2 = st.columns(2)
with col1:
    st.image(f'{webtoon_align_path}/{image_align}', width=256)
with col2:
    st.image(cropped_img, width=256)   


col1, col2, col3 = st.columns(3)
button = col2.button('save image 😉')

if button:
    os.makedirs(f'{current_path}/{webtoon}/align-final', exist_ok=True)
    st.info(f'{current_path}/{webtoon}/align-final/{fail_webtoon}')
    cropped_img.save(f'{current_path}/{webtoon}/align-final/{fail_webtoon}')