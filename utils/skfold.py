import json
import numpy as np
import pandas as pd
import argparse
import os

from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold

DATA_PATH = "../dataset"
IMG_PATH = os.path.join(DATA_PATH, "train")
JSON_PATH = os.path.join(DATA_PATH, "train.json")


def application(args):
    with open(JSON_PATH, "r+", encoding="utf-8") as f:
        train_data = json.load(f)
        images = train_data["images"]
        annotations = train_data["annotations"]
    # 학습데이터 COCO JSON 데이터프레임 생성
    images_df = pd.DataFrame.from_dict(images)
    annotations_df = pd.DataFrame.from_dict(annotations)
    annotations = annotations_df[annotations_df["area"] >= args.filter].to_dict("records")
    x = np.ones((len(annotations), 1))
    y = np.array([item["category_id"] for item in annotations])
    groups = np.array([item["image_id"] for item in annotations])

    cross_val = StratifiedGroupKFold(n_splits=args.n_split, shuffle=True, random_state=1)
    dst_path = args.path

    if not os.path.exists(dst_path):
        os.makedirs(dst_path, exist_ok=True)

    # Stratified Group K Fold - Split dataset by arg.n_split
    for idx, (train_idx, val_idx) in tqdm(enumerate(cross_val.split(x, y, groups)), total=args.n_split):
        train_dict = dict()
        val_dict = dict()
        for item in ["info", "licenses", "categories"]:
            train_dict[item] = train_data[item]
            val_dict[item] = train_data[item]
        train_img_ids = list(set(groups[train_idx]))
        val_img_ids = list(set(groups[val_idx]))
        train_dict["images"] = images_df[images_df["id"].isin(train_img_ids)].to_dict("records")
        train_dict["annotations"] = annotations_df[annotations_df["image_id"].isin(train_img_ids)].to_dict("records")
        val_dict["images"] = images_df[images_df["id"].isin(val_img_ids)].to_dict("records")
        val_dict["annotations"] = annotations_df[annotations_df["image_id"].isin(val_img_ids)].to_dict("records")

        # save splitted train/validataion dataset as separate json files
        dst_train = os.path.join(dst_path, f"train_cv_{idx + 1}.json")
        dst_val = os.path.join(dst_path, f"val_cv_{idx + 1}.json")

        with open(dst_train, "w", encoding="utf-8") as f:
            json.dump(train_dict, f)
        with open(dst_val, "w", encoding="utf-8") as f:
            json.dump(val_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', '-f', type=int, default=0)  # bounding box의 area가 filter값 이상인 것만 사용
    parser.add_argument('--n_split', '-n', type=int, default=5) # 몇개의 데이터셋으로 나눌지 결정 (N-Fold)
    parser.add_argument('--path', '-p', type=str, default=f'{DATA_PATH}/kfold/coco/filter_{parser.parse_args().filter}/nsplit{parser.parse_args().n_split}') # json 파일 경로가 저장될 위치
    arg = parser.parse_args()
    application(arg)
