import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from .yolo_config import write_data_yaml


def train(
    data_yaml: Path,
    model_name: str = "yolov8n-seg.pt",
    epochs: int = 50,
    imgsz: int = 512,
    batch: int = 8,
    device: str = "cpu",
    project: Path = Path("models"),
    name: str = "field-seg",
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
        default=50,
        help="Eğitim epoch sayısı",
    )
    train_p.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help="Girdi çözünürlüğü",
    )
    train_p.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size",
    )
    train_p.add_argument(
        "--model",
        type=str,
        default="yolov8n-seg.pt",
        help="Başlangıç segmentasyon modeli (ör: yolov8n-seg.pt)",
    )
    train_p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device (ör: cpu, 0, 0,1)",
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
        )
    elif args.command == "predict":
        predict(
            weights=Path(args.weights),
            source=Path(args.source),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
