import streamlit as st
import face_recognition
import numpy as np
from PIL import Image

st.set_page_config(page_title="Face AI Demo", layout="centered")
st.title("🔍 FaceCheck Mobile AI Demo")
st.write("Do photos upload karein aur check karein ki dono ek hi insaan hain ya alag!")

file1 = st.file_uploader("Pehli Photo Upload Karein", type=["jpg", "png", "jpeg"], key="1")
file2 = st.file_uploader("Dusri Photo Upload Karein", type=["jpg", "png", "jpeg"], key="2")

if file1 and file2:
    img1 = Image.open(file1).convert('RGB')
    img2 = Image.open(file2).convert('RGB')
    
    st.image([img1, img2], caption=['Photo 1', 'Photo 2'], width=150)
    st.write("🔄 Checking match...")
    
    enc1 = face_recognition.face_encodings(np.array(img1))
    enc2 = face_recognition.face_encodings(np.array(img2))
    
    if len(enc1) > 0 and len(enc2) > 0:
        match = face_recognition.compare_faces([enc1[0]], enc2[0], tolerance=0.6)
        distance = face_recognition.face_distance([enc1[0]], enc2[0])[0]
        confidence = round((1 - distance) * 100, 2)
        
        if match[0]:
            st.success(f"🎉 **Match Found!** Yeh ek hi insaan hain. Match: {confidence}%")
        else:
            st.error(f"❌ **No Match!** Dono alag-alag insaan hain. Match score: {confidence}%")
    else:
        st.warning("⚠️ Kisi ek photo me chehra saaf nahi dikh raha hai. Kripya doosri photo try karein.")
