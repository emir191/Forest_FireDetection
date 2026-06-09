# 🚁 İHA/Drone Tabanlı Akıllı Orman Yangını Tespit ve Erken Uyarı Sistemi (SAHI & YOLOv8)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU%20Accelerated-orange.svg?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?style=for-the-badge&logo=streamlit)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg?style=for-the-badge)

Bu proje, Namık Kemal Üniversitesi Bilgisayar Mühendisliği Bölümü bitirme çalışması kapsamında; İHA/Drone kameralarından alınan geniş alan görüntülerindeki **başlangıç aşamasındaki küçük yangın odaklarını ve uzak dumanları** gerçek zamanlı olarak tespit etmek amacıyla geliştirilmiş yapay zeka destekli bir erken uyarı sistemidir.

---

## 🚀 Öne Çıkan Özellikler

* **SAHI (Slicing Aided Hyper Inference) Entegrasyonu:** Yüksek çözünürlüklü drone görüntülerinde piksel kaybını (downsampling) önlemek amacıyla dinamik dilimleme yöntemi (`get_sliced_prediction`) kullanılmıştır.
* **Zorlu Negatif Öğrenme (Hard Negative):** İlk aşamalardaki sis, bulut ve güneş yansımalarından kaynaklı hatalı alarmları (False Positives) önlemek amacıyla model negatif örneklerle eğitilmiştir.
* **Çoklu Sınıf Algılama:** Sistem, ateş ve duman yapılarını ayrı ayrı kararlı bir şekilde sınıflandırabilir.
* **Akıllı Erken Uyarı Mekanizması:** Toplam güven eşiği aşıldığında dinamik olarak sesli (`winsound.Beep`) ve görsel (HTML/CSS) alarmlar tetiklenir.
* **Kullanıcı Dostu Arayüz:** Fotoğraf, video ve canlı kamera (Webcam/RTSP) kaynaklarını destekleyen gelişmiş Streamlit paneli.

---

## 📸 Arayüz ve Uygulama Görselleri



<img width="910" height="850" alt="Ekran görüntüsü 2026-04-17 175006" src="https://github.com/user-attachments/assets/47c07269-57c8-4e87-bcf1-7c913abe9a5d" />
<img width="1087" height="762" alt="Ekran görüntüsü 2026-05-19 155844" src="https://github.com/user-attachments/assets/506fa654-1201-4b46-9246-dd0ab0dbc693" />


<img width="1056" height="690" alt="sonrasi" src="https://github.com/user-attachments/assets/e30b405b-6f44-410b-bd3e-bd3e13d664dd" />


---

## 📊 Deneysel Sonuçlar ve Performans

Modelin genel performans testleri ile sadece sis içeren zorlu negatif test kümesinden elde edilen akademik metrikler aşağıdaki gibidir:

| Test Senaryosu | Kesinlik (Precision) | Duyarlılık (Recall) | mAP50 | mAP50-95 | Çıkarım Hızı (Kare Başına) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Genel Yangın Testi** | **%90.74** | **%85.34** | **%92.84** | **%60.01** | **2.72 ms** |
| **Zorlu Negatif Testi (Sadece Sis)** | **%63.86** | **%60.94** | **%62.12** | **%31.63** | **2.72 ms** |

* **Hız Performansı:** Toplam ön işleme, çıkarım ve ardışık postprocess (NMS) dahil olmak üzere tek kare analiz süresi ortalama **2.85 ms** olarak ölçülmüştür. Bu performans gerçek zamanlı ($>30$ FPS) video işleme sınırlarını fazlasıyla karşılamaktadır.

---

## ⚙️ Sistem Ayarları & Parametreler

Geliştirilen Streamlit arayüzü üzerinden aşağıdaki parametreler dinamik olarak optimize edilebilmektedir:
* **Minimum Kutu Eşiği (Confidence):** `0.35`
* **Kutu Birleşme Eşiği (IoU):** `0.45`
* **SAHI Dilim Boyutu (Slice Size):** `640x640` (Dinamik: 256 - 1024)
* **SAHI Örtüşme Oranı (Overlap Ratio):** `0.2`

---

