import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class COCODataset(Dataset):
    """
    PyTorch Dataset for COCO format data.
    
    """
    def __init__(self, root_dir, annotation_file, transform=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Directory containing the images
            annotation_file (str): Path to COCO annotation JSON file
            transform (callable, optional): Transform to apply to images
        """
        self.root_dir = root_dir
        with open(annotation_file, 'r') as f:
            coco = json.load(f)
        self.transform = transform
        
        # Map image IDs to filenames for quick lookup
        self.img_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
        self.img_ids = list(self.img_id_to_filename.keys())
        
        # Group annotations by image ID for faster access
        self.annotations = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append((ann['bbox'], ann['category_id']))

        # Convert COCO category IDs to sequential indices for model compatibility
        self.category_id_to_index = {cat['id']: idx for idx, cat in enumerate(coco['categories'])}
        # Store category names for visualization
        self.id_to_label = {cat['id']: cat['name'] for cat in coco['categories']}
    
    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.img_ids)
    
    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        
        Args:
            index (int): Index
            
        Returns:
            tuple: (image, target) where target is a dictionary of annotations
        """
        img_id = self.img_ids[index]
        filename = self.img_id_to_filename[img_id]
        
        # Load and convert image
        img = Image.open(os.path.join(self.root_dir, filename)).convert('RGB')
        
        # Process annotations for this image
        boxes, labels = [], []
        for bbox, cat_id in self.annotations.get(img_id, []):
            # Convert [x, y, width, height] to [x1, y1, x2, y2] format
            x, y, w, h = bbox
            boxes.append([x, y, x+w, y+h])
            # Convert category ID to model index
            labels.append(self.category_id_to_index[cat_id])
        
        # Create target dict with PyTorch tensors
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id])
        }
        
        # Apply any transformations
        if self.transform:
            img = self.transform(img)
        
        return img, target
    
    def get_category_name(self, idx):
        """
        Convert model index back to category name.
        
        Args:
            idx (int): Model index for category
            
        Returns:
            str: Category name
        """
        # Find original category ID that maps to this index
        for cat_id, index in self.category_id_to_index.items():
            if index == idx:
                return self.id_to_label[cat_id]
        return "unknown"