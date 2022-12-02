# Utils Package
`utils` 폴더 내 파일들을 설명하고 사용법을 정의하는 문서입니다.
- `requirements.txt` : Stratified Group K-Fold를 활용한 데이터셋 분할 기능과, Convert2YOLO를 사용하기 위한 의존성 패키지 정의 파일
    ```shell
    $ python -m pip install -r requirements.txt
    ```
- `skfold.py` : Stratified Group K-Fold를 위한 데이터셋 분할 프로그램 파일
    ```shell
    $ python skfold.py [-n NUMBER_OF_FOLDS_TO_SPLIT] [-f AREA_SIZE_FOR_FILTERING] [-p DESTINATION_PATH]
    ```
  -  `-n` (or `--n_split`) : 몇 개의 Fold로 분리할 것인지 설정 (기본값 5, int)
  -  `-f` (or `--filter`) : COCO Dataset 내 Annotations에 포함된 Bounding Box 중 특정 수치 이하의 값을 제거하고 싶을 때 사용 (기본값 0, int)
  -  `-p` (or `--path`) : 생성된 K-Fold 데이터셋을 저장할 경로 설정 (기본값 /dataset/kfold/coco/filter_{FILTER_SIZE}/nsplit{N_SPLIT_VALUE}, str)
- `skfold.bat`, `skfold.sh` : Windows 환경(`.bat`) 또는 Mac/Linux 환경(`.sh`)에서 다수의 K-Fold를 적용하고자 할 때 사용하는 스크립트 명령어
- `kfold_verifier.ipynb` : 생성된 K-Fold Dataset이 어떤 비율로 생성되었는지 확인하기 위한 Jupyter Notebook 파일

## convert2Yolo Package
- `Format.py` : 데이터셋 타입 변환을 위한 Python Class 파일
- `coco2yolo.py` : 데이터셋 타입 변환을 위한 프로그램 파일
  ```shell
  $ python coco2yolo.py [--dataset TYPE_OF_DATASET] [-n TARGET_FOLD_NUMBER] [-f TARGET_FOLD_NUMBER] [-k 1] [-t TRAIN_OR_VAL]
  ```
  - `--datasets` : 데이터셋 타입 (기본값 COCO, str)
  - `--img_path` : 이미지 파일 경로 (기본값 /dataset/train/, str)
  - `--n_split`(or `-n`) : 변환할 데이터셋의 `n_split` 번호 (기본값 5, int)
  - `--filter`(or `-f`) : 변환할 데이터셋의 `filter` 사이즈 (기본값 0, int)
  - `--kfold`(or `-k`) : 변환할 데이터셋의 `fold` 번호 (기본값 1, int)
  - `--type`(or `-t`) : 변환할 데이터셋의 종류(`train`/`val`) (기본값 train, str)
  - `--label`(or `-l`) : 변환할 원본 데이터셋 경로 (기본값 /dataset/kfold/{TYPE_OF_DATASET}}/filter_{FILTER_SIZE}/nsplit{NUM_SPLIT}/{TYPE}_cv_{NUM_FOLD}.json, str)
  - `--convert_output_path`(or `-c`) : 저장 경로 (기본값 /dataset/kfold/yolo/filter_{FILTER_SIZE}/nsplit{NUM_SPLIT}/{TYPE}_cv_{NUM_FOLD}, str)
  - `--img_type` : 데이터셋 이미지 확장자 (기본값 .jpg, str)
  - `--manifest_path` : 생성된 이미지 파일명 manifest 파일 경로 (기본값 ./filter_{FILTER_SIZE}/nsplit{NUM_SPLIT}/, str)
  - `--cls_list_file` : 데이터셋의 클래스를 나열한 클래스 리스트 텍스트 파일 경로 (기본값 names.txt, str)
  
- `names.txt` : 클래스 리스트 텍스트 파일
- `c2y.bat`, `c2y.sh` : Windows 환경(`.bat`) 또는 Mac/Linux 환경(`.sh`)에서 다수의 변환을 일괄적으로 실행하고자 할 때 사용하는 스크립트 명령어