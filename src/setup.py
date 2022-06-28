# Requirements
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import logging as log

# Method to load data and compute 
def setup_data(model_config:str):

    #
    # Part I: Metadata
    #

    # Read config
    with open(model_config, 'r') as f:
        model_dct = json.load(f)

    # Training metadata
    with open('input/MSCOCO/annotations/captions_train2017.json', 'r') as elem:
        ann_dct = json.load(elem)
        train_caption2id = pd.DataFrame(ann_dct['annotations'])
        train_caption2id['image_id'] = train_caption2id['image_id'].apply(lambda x: 'input/MSCOCO/images/train2017/'+ str(x).zfill(12)+'.jpg')
        train_df = train_caption2id.groupby('image_id')['caption'].apply(lambda x: '||'.join(x).split('||')).reset_index()
        del ann_dct, train_caption2id
    with open('input/MSCOCO/annotations/instances_train2017.json', 'r') as elem:
        inst_dct = json.load(elem)
        categories_df = pd.DataFrame(inst_dct['categories'])
        train_instance2id = pd.DataFrame(inst_dct['annotations'])[['image_id', 'category_id']].drop_duplicates()
        train_instance2id['image_id'] = train_instance2id['image_id'].apply(lambda x: 'input/MSCOCO/images/train2017/'+str(x).zfill(12)+'.jpg')
        train_instance2id['category_id'] = train_instance2id['category_id'].astype('str')
        train_instance2id = train_instance2id.groupby('image_id')['category_id'].apply(lambda x: ','.join(x).split(',')).reset_index()
        train_data = train_df.merge(train_instance2id, on='image_id', how='inner')
        del inst_dct, train_df, train_instance2id
    # Validation metadata
    with open('input/MSCOCO/annotations/captions_val2017.json', 'r') as elem:
        ann_dct = json.load(elem)
        val_caption2id = pd.DataFrame(ann_dct['annotations'])
        val_caption2id['image_id'] = val_caption2id['image_id'].apply(lambda x: 'input/MSCOCO/images/val2017/'+ str(x).zfill(12)+'.jpg')
        val_df = val_caption2id.groupby('image_id')['caption'].apply(lambda x: '||'.join(x).split('||')).reset_index()
        del ann_dct, val_caption2id
    with open('input/MSCOCO/annotations/instances_val2017.json', 'r') as elem:
        inst_dct = json.load(elem)
        val_instance2id = pd.DataFrame(inst_dct['annotations'])[['image_id', 'category_id']].drop_duplicates()
        val_instance2id['image_id'] = val_instance2id['image_id'].apply(lambda x: 'input/MSCOCO/images/val2017/'+str(x).zfill(12)+'.jpg')
        val_instance2id['category_id'] = val_instance2id['category_id'].astype('str')
        val_instance2id = val_instance2id.groupby('image_id')['category_id'].apply(lambda x: ','.join(x).split(',')).reset_index()
        val_data = val_df.merge(val_instance2id, on='image_id', how='inner')
        del inst_dct, val_df, val_instance2id
    # Get classes dictionaries
    ids = categories_df['id'].values
    names = categories_df['name'].values
    idx2key = {k:v for k,v in zip(ids, range(len(ids)))}
    idx2name = {k:v for k,v in zip(ids, names)}
    name2key = {idx2name[k]:v for k,v in idx2key.items()}
    key2name = {v:k for k,v in name2key.items()}
    train_data['category_id'] = train_data['category_id'].apply(lambda x: [idx2key[int(elem)] for elem in x if int(elem) in ids])
    train_data = train_data.loc[train_data['category_id'].apply(lambda x: len(x)>0),:].reset_index(drop=True)
    val_data['category_id'] = val_data['category_id'].apply(lambda x: [idx2key[int(elem)] for elem in x if int(elem) in ids])
    val_data = val_data.loc[val_data['category_id'].apply(lambda x: len(x)>0),:].reset_index(drop=True)

    #
    # Part II: Max length
    #

    # Estimate max length
    length_list = []
    tokenizer = AutoTokenizer.from_pretrained(model_dct['text_backbone'])
    for elem in tqdm(train_data.explode('caption')['caption']):
        length_list.append(len(tokenizer(elem).input_ids))
    max_length_corpus = int(np.quantile(length_list, .995))
    log.info(f"Recommended maximum length: {max_length_corpus}")

    #
    # Part III: Write updated config
    #

    with open(model_config, 'w') as f:
        model_dct['max_length_corpus'] = max_length_corpus
        json.dump(model_dct, f)


    # Exit
    return train_data, val_data, key2name