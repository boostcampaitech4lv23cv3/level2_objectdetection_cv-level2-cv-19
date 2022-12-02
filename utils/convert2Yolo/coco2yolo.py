import os
import argparse
from Format import VOC, COCO, UDACITY, KITTI, YOLO


DATA_PATH = "../../dataset"
parser = argparse.ArgumentParser(description='label Converting example.')
parser.add_argument('--datasets', type=str, default="COCO", help='type of datasets')
parser.add_argument('--img_path', type=str, default=f"{DATA_PATH}/train/", help='directory of image folder')
parser.add_argument('--n_split', '-n', type=int, default=5)
parser.add_argument('--filter', '-f', type=int, default=0)
parser.add_argument('--type', '-t', type=str, default="train")
parser.add_argument('--kfold', '-k', type=int, default=1)
parser.add_argument('--label', '-l', type=str, default=f"{DATA_PATH}/kfold/coco/filter_{parser.parse_args().filter}/nsplit{parser.parse_args().n_split}/{parser.parse_args().type}_cv_{parser.parse_args().kfold}.json", help='directory of label folder or label file path')
parser.add_argument('--convert_output_path', '-c', type=str, default=f"{DATA_PATH}/kfold/yolo/filter_{parser.parse_args().filter}/nsplit{parser.parse_args().n_split}/{parser.parse_args().type}_cv_{parser.parse_args().kfold}", help='directory of label folder')
parser.add_argument('--img_type', type=str, default=".jpg", help='type of image')
parser.add_argument('--manifest_path', type=str, help='directory of manipast file', default="./")
parser.add_argument('--cls_list_file', type=str, help='directory of *.names file', default="names.txt")

args = parser.parse_args()


def main(config):

    if config["datasets"] == "VOC":
        voc = VOC()
        yolo = YOLO(os.path.abspath(config["cls_list"]))

        flag, data = voc.parse(config["label"])

        if flag == True:

            flag, data = yolo.generate(data)
            if flag == True:
                flag, data = yolo.save(data, config["output_path"], config["img_path"],
                                       config["img_type"], config["manifest_path"])

                if flag == False:
                    print("Saving Result : {}, msg : {}".format(flag, data))

            else:
                print("YOLO Generating Result : {}, msg : {}".format(flag, data))

        else:
            print("VOC Parsing Result : {}, msg : {}".format(flag, data))

    elif config["datasets"] == "COCO":
        coco = COCO()

        flag, data, cls_hierarchy = coco.parse(
            config["label"], config["img_path"])
        yolo = YOLO(os.path.abspath(
            config["cls_list"]), cls_hierarchy=cls_hierarchy)

        if flag == True:
            flag, data = yolo.generate(data)

            if flag == True:
                flag, data = yolo.save(data, config["output_path"], config["img_path"],
                                       config["img_type"], config["manifest_path"])

                if flag == False:
                    print("Saving Result : {}, msg : {}".format(flag, data))

            else:
                print("YOLO Generating Result : {}, msg : {}".format(flag, data))

        else:
            print("COCO Parsing Result : {}, msg : {}".format(flag, data))

    elif config["datasets"] == "UDACITY":
        udacity = UDACITY()
        yolo = YOLO(os.path.abspath(config["cls_list"]))

        flag, data = udacity.parse(config["label"])

        if flag == True:
            flag, data = yolo.generate(data)

            if flag == True:
                flag, data = yolo.save(data, config["output_path"], config["img_path"],
                                       config["img_type"], config["manifest_path"])

                if flag == False:
                    print("Saving Result : {}, msg : {}".format(flag, data))

            else:
                print("UDACITY Generating Result : {}, msg : {}".format(flag, data))

        else:
            print("COCO Parsing Result : {}, msg : {}".format(flag, data))

    elif config["datasets"] == "KITTI":
        kitti = KITTI()
        yolo = YOLO(os.path.abspath(config["cls_list"]))

        flag, data = kitti.parse(
            config["label"], config["img_path"], img_type=config["img_type"])

        if flag == True:
            flag, data = yolo.generate(data)

            if flag == True:
                flag, data = yolo.save(data, config["output_path"], config["img_path"],
                                       config["img_type"], config["manifest_path"])

                if flag == False:
                    print("Saving Result : {}, msg : {}".format(flag, data))

            else:
                print("YOLO Generating Result : {}, msg : {}".format(flag, data))

        else:
            print("KITTI Parsing Result : {}, msg : {}".format(flag, data))

    else:
        print("Unkwon Datasets")


if __name__ == '__main__':

    config = {
        "datasets": args.datasets,
        "img_path": args.img_path,
        "label": args.label,
        "img_type": args.img_type,
        "manifest_path": args.manifest_path,
        "output_path": args.convert_output_path,
        "cls_list": args.cls_list_file,
    }

    if not os.path.exists(config['output_path']):
        os.makedirs(config['output_path'])

    main(config)