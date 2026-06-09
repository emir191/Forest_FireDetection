import streamlit as st
import cv2
from PIL import Image
import time
import numpy as np
import tempfile
import threading
import winsound


from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions

st.set_page_config(page_title="Gelişmiş Yangın Tespit Sistemi", layout="wide")


@st.cache_resource
def load_sahi_model():
    import torch
    if not torch.cuda.is_available():
        st.error("⚠️ DİKKAT: PyTorch ekran kartını (GPU) göremiyor! Sistem İşlemci (CPU) ile devam ediyor.")
    else:
        st.success(f"🚀 GPU Aktif: {torch.cuda.get_device_name(0)}")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='best_akilliavci.pt',
        confidence_threshold=0.1, 
        device="cuda:0" 
    )
    return detection_model

model = load_sahi_model()


def process_frame(img_array, conf_thresh, slice_s, overlap_r, iou_v, total_conf_thresh):
    
    img_array = cv2.resize(img_array, (1280, 720))
    model.confidence_threshold = conf_thresh
    
    start_time = time.time() 
    
    
    result = get_sliced_prediction(
        img_array,
        model,
        slice_height=slice_s,
        slice_width=slice_s,
        overlap_height_ratio=overlap_r,
        overlap_width_ratio=overlap_r,
        postprocess_type="NMS",
        postprocess_match_metric="IOU",
        postprocess_match_threshold=iou_v
    )
    
    end_time = time.time()  
    processing_duration = end_time - start_time 
    
    
    print(f"Tek kare analiz süresi: {processing_duration:.3f} saniye")
    
    total_conf = 0.0
    if result.object_prediction_list:
        for obj in result.object_prediction_list:
            total_conf += float(obj.score.value)
            
    
    annotated_frame = visualize_object_predictions(
        img_array.copy(),
        object_prediction_list=result.object_prediction_list,
        rect_th=2, 
        text_size=0.5,
        text_th=1
    )["image"]
    
    is_fire = total_conf >= total_conf_thresh
    
   
    st.sidebar.write(f"⏱️ Son Analiz Süresi: {processing_duration:.3f} sn")
    
    return annotated_frame, total_conf, is_fire


# ==========================================
st.title("🚁 İHA/Drone Yangın Takip Sistemi (SAHI)")

st.sidebar.header("📡 Kaynak Seçimi")
source_radio = st.sidebar.radio("Analiz Türü:", ["Görsel (Fotoğraf)", "Video Yükle", "Canlı Kamera"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Ayarlar")
camera_name_input = st.sidebar.text_input("Kamera/Kaynak Adı", "İstasyon 1")

analysis_interval = st.sidebar.slider("Analiz Sıklığı (Saniyede 1)", 1, 10, 2) 
box_conf_threshold = st.sidebar.slider("Minimum Kutu Eşiği", 0.05, 1.0, 0.35)
total_conf_threshold = st.sidebar.slider("Toplam Güven Eşiği (Alarm)", 0.1, 5.0, 0.55)
iou_val = st.sidebar.slider("Kutu Birleşme Eşiği (IoU)", 0.1, 1.0, 0.45)
slice_size = st.sidebar.slider("SAHI Dilim Boyutu", 256, 1024, 640, step=128)
overlap_ratio = st.sidebar.slider("SAHI Örtüşme Oranı", 0.0, 0.5, 0.2, step=0.1)

warning_placeholder = st.empty()
frame_placeholder = st.empty()
status_text = st.empty()


if "vid_cap" not in st.session_state:
    st.session_state.vid_cap = None
if "last_alarm_time" not in st.session_state:
    st.session_state.last_alarm_time = 0


def trigger_alarm(total_conf):
    current_time = time.time()
    warning_html = f"""
    <div style="background-color: #ff4b4b; color: white; padding: 15px; 
                border-radius: 8px; text-align: center; font-size: 24px; 
                font-weight: bold; margin-bottom: 20px; 
                box-shadow: 0px 4px 6px rgba(0,0,0,0.4);">
        🚨 DİKKAT: {camera_name_input} YANGIN ALGILADI! (Skor: {total_conf:.2f}) 🚨
    </div>
    """
    warning_placeholder.markdown(warning_html, unsafe_allow_html=True)
    
    if current_time - st.session_state.last_alarm_time >= 3.0:
        threading.Thread(target=winsound.Beep, args=(2500, 1000), daemon=True).start()
        st.session_state.last_alarm_time = current_time


if source_radio == "Görsel (Fotoğraf)":
    uploaded_files = st.file_uploader("Fotoğraf yükleyin:", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files and st.button("▶️ Analizi Başlat"):
        for file in uploaded_files:
            image = Image.open(file).convert("RGB")
            img_array = np.array(image)
            
            annotated_frame, total_conf, is_fire = process_frame(
                img_array, box_conf_threshold, slice_size, overlap_ratio, iou_val, total_conf_threshold
            )
            
            if is_fire: trigger_alarm(total_conf)
            else: warning_placeholder.empty()
                
            frame_placeholder.image(annotated_frame, use_container_width=True)
            time.sleep(0.1)
        status_text.success("Analiz bitti.")


elif source_radio == "Video Yükle":
    uploaded_video = st.file_uploader("Video yükleyin:", type=["mp4", "avi", "mov"])
    if uploaded_video and st.button("▶️ Videoyu Analiz Et"):
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 24 
        
        frames_to_skip = int(fps * analysis_interval)
        frame_counter = 0
        
        status_text.info(f"Video {analysis_interval} saniyelik aralıklarla analiz ediliyor...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_counter += 1
            
            if frame_counter % frames_to_skip == 0 or frame_counter == 1:
                img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated_frame, total_conf, is_fire = process_frame(
                    img_array, box_conf_threshold, slice_size, overlap_ratio, iou_val, total_conf_threshold
                )
                
                if is_fire: trigger_alarm(total_conf)
                else: warning_placeholder.empty()
                    
                frame_placeholder.image(annotated_frame, use_container_width=True)
                
            time.sleep(0.01)
            
        cap.release()
        status_text.success("Video analizi tamamlandı.")


elif source_radio == "Canlı Kamera":
    run_system = st.checkbox("🔴 Kamerayı Başlat / Durdur")
    
    if run_system:
        if st.session_state.vid_cap is None:
            st.session_state.vid_cap = cv2.VideoCapture(0)
            st.session_state.vid_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        cap = st.session_state.vid_cap
        if not cap.isOpened():
            st.error("Kamera açılamadı!")
        else:
            status_text.info(f"Sistem devrede. Her {analysis_interval} saniyede bir analiz yapılıyor.")
            
            last_analysis_time = 0
            
            while run_system:
                ret, frame = cap.read()
                if not ret: break

                current_time = time.time()
                
                if current_time - last_analysis_time >= analysis_interval:
                    img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    annotated_frame, total_conf, is_fire = process_frame(
                        img_array, box_conf_threshold, slice_size, overlap_ratio, iou_val, total_conf_threshold
                    )
                    
                    if is_fire: trigger_alarm(total_conf)
                    else: warning_placeholder.empty()
                        
                    frame_placeholder.image(annotated_frame, use_container_width=True)
                    last_analysis_time = current_time
                
                time.sleep(0.05)
                if not run_system: break
    else:
        if st.session_state.vid_cap is not None:
            st.session_state.vid_cap.release()
            st.session_state.vid_cap = None
        warning_placeholder.empty()
        frame_placeholder.empty()
        st.info("Kamerayı açmak için kutucuğu işaretleyin.")
