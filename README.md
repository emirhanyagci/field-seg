# Field Segmentation - Tarım Alanı Segmentasyonu

YOLOv8 kullanarak tarım alanlarını segmentasyon yöntemiyle tespit eden ve maskeleyen bir model projesi.

## Özellikler

- ✅ YOLOv8 Segmentation modeli ile eğitim
- ✅ Pixel-level maskeleme (sadece alan seçimi, çerçeve değil)
- ✅ Temiz ve modüler kod yapısı
- ✅ Eğitim ve inference scriptleri
- ✅ Batch processing desteği
- ✅ Görselleştirme ve analiz araçları

## Kurulum

1. Repository'yi klonlayın:
```bash
git clone <repository-url>
cd field-seg
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

## Veri Yapısı

Proje şu şekilde organize edilmiştir:

```
field-seg/
├── data/
│   └── field/
│       ├── images/     # Görüntü dosyaları (.tif)
│       └── labels/     # YOLO format etiket dosyaları (.txt)
├── field-seg.yaml      # Veri konfigürasyon dosyası
├── src/
│   ├── main.py         # Ana CLI arayüzü
│   ├── train.py        # Eğitim scripti
│   ├── inference.py    # Inference scripti
│   ├── config.py       # Konfigürasyon ayarları
│   └── utils.py        # Yardımcı fonksiyonlar
└── requirements.txt
```

## Kullanım

### Eğitim

Modeli eğitmek için:

```bash
# Temel kullanım
python src/main.py train --epochs 100 --batch-size 16

# Farklı model boyutu ile
python src/main.py train --model-size s --epochs 150 --batch-size 32

# Özel ayarlarla
python src/main.py train \
    --epochs 100 \
    --batch-size 16 \
    --imgsz 640 \
    --workers 8 \
    --device cuda \
    --name my_experiment
```

**Model Boyutları:**
- `n` (nano) - En hızlı, en küçük
- `s` (small) - Dengeli
- `m` (medium) - Daha iyi doğruluk
- `l` (large) - Yüksek doğruluk
- `x` (xlarge) - En yüksek doğruluk

Eğitim sonrası model `runs/field_segmentation/weights/best.pt` konumuna kaydedilir.

### Inference (Tahmin)

Tek bir görüntü üzerinde tahmin yapmak için:

```bash
python src/main.py predict \
    --source data/test/image.tif \
    --output predictions \
    --model runs/field_segmentation/weights/best.pt
```

Bir dizindeki tüm görüntüleri işlemek için:

```bash
python src/main.py predict \
    --source data/test/images \
    --output predictions \
    --conf 0.25 \
    --iou 0.45 \
    --save-mask
```

**Inference Parametreleri:**
- `--source`: Girdi görüntü veya dizin yolu
- `--output`: Sonuçların kaydedileceği dizin
- `--model`: Eğitilmiş model yolu (varsayılan: `best.pt`)
- `--conf`: Güven eşiği (0-1, varsayılan: 0.25)
- `--iou`: NMS IoU eşiği (varsayılan: 0.45)
- `--save-mask`: Maske dosyalarını ayrı olarak kaydet
- `--mask-alpha`: Maske şeffaflığı (0-1, varsayılan: 0.5)

## Kod Yapısı

### Modüller

- **`config.py`**: Model ve inference konfigürasyon sınıfları
- **`train.py`**: YOLOv8 model eğitimi
- **`inference.py`**: Tahmin ve görselleştirme
- **`utils.py`**: Yardımcı fonksiyonlar (maskeleme, görselleştirme, vb.)
- **`main.py`**: Komut satırı arayüzü

### Best Practices

1. **Modüler Yapı**: Her modül belirli bir sorumluluğa sahip
2. **Type Hints**: Fonksiyonlarda tip ipuçları kullanıldı
3. **Logging**: Tüm işlemler loglanıyor
4. **Error Handling**: Hata durumları ele alınıyor
5. **Documentation**: Fonksiyonlar dokümante edildi
6. **Configuration Management**: Ayarlar merkezi olarak yönetiliyor

## Çıktılar

Inference sonrası şu dosyalar oluşturulur:

- `*_result.jpg`: Orijinal görüntü üzerine maskeleme ve etiketler
- `*_mask.png`: Sadece segmentasyon maskesi (eğer `--save-mask` kullanıldıysa)

## Performans İpuçları

1. **GPU Kullanımı**: CUDA destekli GPU varsa `--device cuda` kullanın
2. **Batch Size**: GPU belleğinize göre batch size'ı ayarlayın
3. **Image Size**: Daha yüksek çözünürlük için `--imgsz` değerini artırın (daha yavaş)
4. **Model Size**: Daha hızlı inference için küçük model (`n` veya `s`) kullanın

## Lisans

Bu proje açık kaynaklıdır.

