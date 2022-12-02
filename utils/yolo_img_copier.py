import os
import shutil
from tqdm import tqdm

dataset_path = "/".join([os.path.dirname(os.path.realpath(__file__)), "../dataset"])
yolo_path = "/".join([dataset_path, "kfold", "yolo"])

for dir1 in tqdm(os.listdir(yolo_path)):
    filter_path = "/".join([yolo_path, dir1])
    for dir2 in os.listdir(filter_path):
        nsplit_path = "/".join([filter_path, dir2])
        for dir3 in os.listdir(nsplit_path):
            batch_path = "/".join([nsplit_path, dir3])
            for txt_filename in tqdm(os.listdir(batch_path)):
                fn, ext = txt_filename.split(".")
                req_filename = fn + ".jpg"
                shutil.copy(os.path.join(dataset_path, "train", req_filename), os.path.join(batch_path, req_filename))