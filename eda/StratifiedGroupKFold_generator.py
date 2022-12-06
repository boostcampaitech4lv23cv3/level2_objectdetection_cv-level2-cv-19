import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# load json annotation = {dataset file 경로}
annotation = './dataset/train.json'

with open(annotation) as f: 
    data = json.load(f)
    info = data['info']
    licenses = data['licenses']
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    
var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]
X = np.ones((len(data['annotations']),1))
y = np.array([v[1] for v in var])
groups = np.array([v[0] for v in var])
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=411)

for i,(train_idx,val_idx) in enumerate(cv.split(X,y,groups)):
    with open(f'./dataset/train_fold{i}.json','w',encoding='utf-8') as f:
        json.dump({'info':info,'licenses':licenses,'images':[images[j] for j in np.unique(groups[train_idx])],'categories':categories,'annotations':[annotations[k] for k in train_idx] },f,indent=4)
    with open(f'./dataset/val_fold{i}.json','w',encoding='utf-8') as f:
        json.dump({'info':info,'licenses':licenses,'images':[images[j] for j in np.unique(groups[val_idx])],'categories':categories,'annotations':[annotations[k] for k in val_idx] },f,indent=4)