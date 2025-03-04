import fiftyone as fo
import fiftyone.zoo as foz

# Download COCO sample
dataset = foz.load_zoo_dataset("coco-2017", split="validation", max_samples=200)

# Create random 70/15/15 split
train_view = dataset.take(140)
val_view = dataset.exclude(train_view).take(30)
test_view = dataset.exclude(train_view).exclude(val_view)

# Export views as separate datasets
train_dataset = train_view.clone()
train_dataset.name = "coco_train"

val_dataset = val_view.clone()
val_dataset.name = "coco_val"

test_dataset = test_view.clone()
test_dataset.name = "coco_test"

# Optional: Export to disk in COCO format
train_dataset.export(
    export_dir="./code/dataset/coco_train",
    dataset_type=fo.types.COCODetectionDataset
)

val_dataset.export(
    export_dir="./code/dataset/coco_val",
    dataset_type=fo.types.COCODetectionDataset
)

test_dataset.export(
    export_dir="./code/dataset/coco_test",
    dataset_type=fo.types.COCODetectionDataset
)