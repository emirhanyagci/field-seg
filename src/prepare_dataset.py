import argparse
import random
from pathlib import Path
from typing import List, Tuple


def _match_image_label_pairs(
    images_dir: Path, labels_dir: Path
) -> List[Tuple[Path, Path]]:
    """
    Eşleşen görüntü (.tif) ve label (.txt) dosyalarını bul.
    İsimler uzantı hariç birebir aynı varsayılır.
    """
    image_files = sorted(images_dir.glob("*.tif"))
    pairs: List[Tuple[Path, Path]] = []

    for img in image_files:
        label = labels_dir / f"{img.stem}.txt"
        if label.exists():
            pairs.append((img, label))

    return pairs


def split_dataset(
    pairs: List[Tuple[Path, Path]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)

    val_size = int(len(shuffled) * val_ratio)
    val_pairs = shuffled[:val_size]
    train_pairs = shuffled[val_size:]
    return train_pairs, val_pairs


def copy_pairs(pairs: List[Tuple[Path, Path]], img_out: Path, lbl_out: Path) -> None:
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img, lbl in pairs:
        img_dest = img_out / img.name
        lbl_dest = lbl_out / lbl.name
        if not img_dest.exists():
            img_dest.write_bytes(img.read_bytes())
        if not lbl_dest.exists():
            lbl_dest.write_bytes(lbl.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO parsel segmentasyonu için /data/field -> /dataset train/val ayrımı"
    )
    parser.add_argument(
        "--source-images",
        type=str,
        default=str(Path("data/field/images")),
        help="Ham görüntü klasörü (tif).",
    )
    parser.add_argument(
        "--source-labels",
        type=str,
        default=str(Path("data/field/labels")),
        help="Ham label klasörü (txt, YOLO polygon format).",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(Path("dataset")),
        help="Çıktı dataset kök klasörü (images/train, images/val, labels/train, labels/val).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Doğrulama oranı (0-1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Rastgelelik için seed.",
    )

    args = parser.parse_args()

    images_dir = Path(args.source_images)
    labels_dir = Path(args.source_labels)
    dataset_root = Path(args.dataset_root)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Görüntü klasörü bulunamadı: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Label klasörü bulunamadı: {labels_dir}")

    pairs = _match_image_label_pairs(images_dir, labels_dir)
    if not pairs:
        raise RuntimeError("Eşleşen görüntü / label çifti bulunamadı.")

    train_pairs, val_pairs = split_dataset(pairs, args.val_ratio, args.seed)

    images_train = dataset_root / "images" / "train"
    images_val = dataset_root / "images" / "val"
    labels_train = dataset_root / "labels" / "train"
    labels_val = dataset_root / "labels" / "val"

    copy_pairs(train_pairs, images_train, labels_train)
    copy_pairs(val_pairs, images_val, labels_val)

    print(f"Toplam eşleşen çift: {len(pairs)}")
    print(f"Train: {len(train_pairs)}  |  Val: {len(val_pairs)}")
    print(f"Dataset kökü: {dataset_root.resolve()}")


if __name__ == "__main__":
    main()
