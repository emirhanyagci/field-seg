import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

# Windows uyumluluğu için import
try:
    from .yolo_config import write_data_yaml
except ImportError:
    from yolo_config import write_data_yaml


def train(
    data_yaml: Path,
    model_name: str = "yolov8s-seg.pt",  # nano yerine small önerilir
    epochs: int = 100,  # daha fazla epoch
    imgsz: int = 640,  # daha yüksek çözünürlük
    batch: int = 16,  # GPU varsa daha büyük batch
    device: str = "cpu",
    project: Path = Path("models"),
    name: str = "field-seg",
    patience: int = 50,  # early stopping
    lr0: float = 0.001,  # daha düşük learning rate
    cos_lr: bool = True,  # cosine learning rate scheduler
    multi_scale: bool = False,  # multi-scale training (yavaşlatır ama iyileştirir)
    cache: bool = False,  # RAM yeterliyse True yapabilirsiniz
    workers: int = 8,  # data loading workers
) -> None:
    project.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_name)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        patience=patience,  # early stopping
        lr0=lr0,  # initial learning rate
        cos_lr=cos_lr,  # cosine LR scheduler
        multi_scale=multi_scale,  # multi-scale training
        cache=cache,  # cache images for faster training
        workers=workers,  # data loading workers
        # Augmentation ayarları (segmentasyon için optimize edilmiş)
        hsv_h=0.015,  # hue augmentation
        hsv_s=0.7,  # saturation augmentation
        hsv_v=0.4,  # value augmentation
        degrees=10.0,  # rotation (parsel için uygun)
        translate=0.1,  # translation
        scale=0.5,  # scaling
        fliplr=0.5,  # horizontal flip
        mosaic=1.0,  # mosaic augmentation
        mixup=0.1,  # mixup augmentation (düşük tutuldu)
        copy_paste=0.1,  # copy-paste augmentation (segmentasyon için iyi)
    )


def predict(
    weights: Path,
    source: Path,
    imgsz: int = 512,
    conf: float = 0.25,
    device: str = "cpu",
    save: bool = True,
    save_dir: Path | None = None,
) -> None:
    model = YOLO(str(weights))
    # YOLO'nun kendi çizdiği kutuları kapatmak için save=False,
    # ardından sadece maskelerin çizildiği görüntüleri biz kaydediyoruz.
    kwargs = dict(imgsz=imgsz, conf=conf, save=False, device=device)
    results = model.predict(source=str(source), **kwargs)

    # Çıktı klasörü
    out_dir = save_dir or Path("runs/segment/predict")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Her sonuç için sadece maskelerin olduğu görüntüyü kaydet
    for r in results:
        img = r.plot(boxes=False)  # kutular kapalı, sadece maskeler
        out_path = out_dir / Path(r.path).name
        cv2.imwrite(str(out_path), img)


def export(
    weights: Path,
    format: str = "onnx",
    imgsz: int = 512,
    device: str = "cpu",
    half: bool = False,
    simplify: bool = True,
    opset: int | None = None,
    workspace: int | None = None,
) -> None:
    """
    Eğitilmiş modeli farklı formatlara export eder.

    Desteklenen formatlar:
    - onnx: ONNX formatı (en yaygın)
    - torchscript: PyTorch TorchScript
    - tensorrt: NVIDIA TensorRT (GPU gerekli)
    - coreml: Apple CoreML (macOS/iOS)
    - engine: TensorRT engine dosyası
    - saved_model: TensorFlow SavedModel
    - pb: TensorFlow protobuf
    - tflite: TensorFlow Lite
    - edgetpu: Google Edge TPU
    - paddle: PaddlePaddle
    - ncnn: NCNN
    - openvino: OpenVINO
    """
    model = YOLO(str(weights))

    export_kwargs = {
        "format": format,
        "imgsz": imgsz,
        "device": device,
        "half": half,
        "simplify": simplify,
    }

    if opset is not None:
        export_kwargs["opset"] = opset
    if workspace is not None:
        export_kwargs["workspace"] = workspace

    # Export işlemi
    exported_path = model.export(**export_kwargs)
    print(f"Model başarıyla export edildi: {exported_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO ile parsel segmentasyonu")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    train_p = subparsers.add_parser("train", help="Model eğit")
    train_p.add_argument(
        "--data-yaml",
        type=str,
        default=str(Path("dataset/field_seg.yaml")),
        help="Ultralytics data.yaml yolu",
    )
    train_p.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Eğitim epoch sayısı (önerilen: 100-200)",
    )
    train_p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Girdi çözünürlüğü (önerilen: 640 veya 832)",
    )
    train_p.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (GPU varsa 16-32, CPU için 4-8)",
    )
    train_p.add_argument(
        "--model",
        type=str,
        default="yolov8s-seg.pt",
        help="Başlangıç segmentasyon modeli (yolov8n/s/m/l/x-seg.pt)",
    )
    train_p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device (ör: cpu, 0, cuda, 0,1)",
    )
    train_p.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (epoch sayısı)",
    )
    train_p.add_argument(
        "--lr0",
        type=float,
        default=0.001,
        help="Initial learning rate (önerilen: 0.001)",
    )
    train_p.add_argument(
        "--cos-lr",
        action="store_true",
        default=True,
        help="Cosine learning rate scheduler kullan",
    )
    train_p.add_argument(
        "--no-cos-lr",
        dest="cos_lr",
        action="store_false",
        help="Cosine LR scheduler'ı kapat",
    )
    train_p.add_argument(
        "--multi-scale",
        action="store_true",
        default=False,
        help="Multi-scale training (yavaşlatır ama iyileştirir)",
    )
    train_p.add_argument(
        "--cache",
        action="store_true",
        default=False,
        help="Görüntüleri RAM'de cache'le (hızlandırır ama RAM gerektirir)",
    )
    train_p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Data loading worker sayısı (varsayılan: 8)",
    )

    # predict
    pred_p = subparsers.add_parser("predict", help="Eğitilmiş model ile tahmin yap")
    pred_p.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Eğitilmiş .pt ağırlık dosyası",
    )
    pred_p.add_argument(
        "--source",
        type=str,
        default=str(Path("dataset/images/val")),
        help="Girdi görüntü yolu veya klasörü",
    )
    pred_p.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help="Girdi çözünürlüğü",
    )
    pred_p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Tahmin güven eşiği",
    )
    pred_p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device (ör: cpu, 0, 0,1)",
    )

    # export
    export_p = subparsers.add_parser("export", help="Eğitilmiş modeli export et")
    export_p.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Eğitilmiş .pt ağırlık dosyası (örn: models/field-seg/weights/best.pt)",
    )
    export_p.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=[
            "onnx",
            "torchscript",
            "tensorrt",
            "coreml",
            "engine",
            "saved_model",
            "pb",
            "tflite",
            "edgetpu",
            "paddle",
            "ncnn",
            "openvino",
        ],
        help="Export formatı (varsayılan: onnx)",
    )
    export_p.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help="Girdi çözünürlüğü",
    )
    export_p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device (ör: cpu, 0, 0,1)",
    )
    export_p.add_argument(
        "--half",
        action="store_true",
        help="FP16 quantizasyon (TensorRT, ONNX için)",
    )
    export_p.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="ONNX modelini sadeleştir (varsayılan: True)",
    )
    export_p.add_argument(
        "--no-simplify",
        dest="simplify",
        action="store_false",
        help="ONNX sadeleştirmeyi kapat",
    )
    export_p.add_argument(
        "--opset",
        type=int,
        default=None,
        help="ONNX opset versiyonu (varsayılan: otomatik)",
    )
    export_p.add_argument(
        "--workspace",
        type=int,
        default=None,
        help="TensorRT workspace boyutu (GB)",
    )

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.command == "train":
        data_yaml = Path(args.data_yaml)
        default_yaml = Path("dataset/field_seg.yaml")

        # Eğer kullanıcı default yolu kullanıyorsa, config'i her seferinde
        # temiz ve doğru formatta yeniden yaz.
        if data_yaml.resolve() == default_yaml.resolve():
            data_yaml = write_data_yaml()
            print(f"data.yaml güncellendi: {data_yaml}")
        elif not data_yaml.exists():
            raise FileNotFoundError(f"Belirtilen data.yaml bulunamadı: {data_yaml}")

        train(
            data_yaml=data_yaml,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            patience=getattr(args, "patience", 50),
            lr0=getattr(args, "lr0", 0.001),
            cos_lr=getattr(args, "cos_lr", True),
            multi_scale=getattr(args, "multi_scale", False),
            cache=getattr(args, "cache", False),
            workers=getattr(args, "workers", 8),
        )
    elif args.command == "predict":
        predict(
            weights=Path(args.weights),
            source=Path(args.source),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
        )
    elif args.command == "export":
        export(
            weights=Path(args.weights),
            format=args.format,
            imgsz=args.imgsz,
            device=args.device,
            half=args.half,
            simplify=args.simplify,
            opset=args.opset,
            workspace=args.workspace,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
