# Requirements
from typing import List, Dict
import pandas as pd
import random
import torch
from transformers import AutoTokenizer
import albumentations as A
import cv2



# Albumentations transform
transforms = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.GaussNoise(p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.PiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),            
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
        A.Resize(224, 224, interpolation=cv2.INTER_AREA),
        A.Normalize(),
    ])


# Torch dataset
class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self,
                 df:pd.DataFrame,
                 text_backbone:str,
                 max_length_corpus:int,
                 key2name:Dict,
                 transforms=transforms,
                 downstream_task:bool=False,
                 ):
        super(CLIPDataset, self).__init__()
        # Parameters
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(text_backbone)
        self.max_length_corpus = max_length_corpus
        self.num_labels = len(key2name)
        self.key2name = key2name
        self.transforms = transforms
        self.downstream_task = downstream_task


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        # Get metadata
        img_path, captions, categories = self.df.loc[idx,'image_id'], self.df.loc[idx,'caption'], self.df.loc[idx,'category_id']
        # Load & preprocess text
        ## Pick up random caption from 5 available
        text = random.choice(captions)
        items = self._collate_HuggingFace(text)
        # Load & preprocess image
        ## Read image
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        ## Apply augmentations
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        items['image'] = torch.from_numpy(image.transpose((2,0,1)))
        # Get label for our problem
        if self.downstream_task:
            items['label'] = torch.LongTensor(self._one_hot_encode(categories))
        return items


    def _one_hot_encode(self, array:List[int]):
        return [1 if idx in array else 0 for idx in range(self.num_labels)]


    def _collate_HuggingFace(self, text):
        # Tokenize text
        tokens = self.tokenizer.encode_plus(text,
                                            max_length=self.max_length_corpus,
                                            padding='max_length',
                                            truncation=True,
                                            return_offsets_mapping=True,
                                            return_tensors='pt',
                                            )
        # End of method
        return {'input_ids': torch.squeeze(tokens['input_ids']), 'attention_mask':torch.squeeze(tokens['attention_mask'])}