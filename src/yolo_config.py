from pathlib import Path


def write_data_yaml(
    dataset_root: Path = Path("dataset"),
    save_path: Path = Path("dataset/field_seg.yaml"),
    class_names: list[str] | None = None,
) -> Path:
    """
    Ultralytics YOLO için basit bir data.yaml üretir.
    Varsayılan olarak tek sınıf: 'parcel'.
    """
    if class_names is None:
        class_names = ["parcel"]

    dataset_root = Path(dataset_root)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Ultralytics mantığı:
    # - path + train/val birleştirilir.
    # - Bu yüzden path = dataset kökü, train/val = 'images/train', 'images/val' olmalı.
    yaml_text = (
        f"path: {dataset_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
    )

    for idx, name in enumerate(class_names):
        yaml_text += f"  {idx}: {name}\n"

    save_path.write_text(yaml_text, encoding="utf-8")
    return save_path


__all__ = ["write_data_yaml"]
