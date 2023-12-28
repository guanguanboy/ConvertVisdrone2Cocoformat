import os
import cv2
from tqdm import tqdm
import json


def test():
    dir=r'E:/Datasets/DroneDatasets/VisDrone2019/VisDrone2019-DET-val/'
    train_dir = os.path.join(dir, "annotations")
    print(train_dir)
    id_num = 0
    categories = [
        {"id": 0, "name": "ignored regions"},
        {"id": 1, "name": "pedestrian"},
        {"id": 2, "name": "people"},
        {"id": 3, "name": "bicycle"},
        {"id": 4, "name": "car"},
        {"id": 5, "name": "van"},
        {"id": 6, "name": "truck"},
        {"id": 7, "name": "tricycle"},
        {"id": 8, "name": "awning-tricycle"},
        {"id": 9, "name": "bus"},
        {"id": 10, "name": "motor"},
        {"id": 11, "name": "others"}
    ]
    images = []
    annotations = []
    #set = os.listdir('./annotations')
    #annotations_path = './annotations'
    #images_path = './images'
    set = os.listdir(train_dir)
    annotations_path = train_dir
    images_path = os.path.join(dir, 'images')
    print()
    for i in tqdm(set):
        print(annotations_path + "/" + i, "r")
        f = open(annotations_path + "/" + i, "r")
        name = i.replace(".txt", "")
        image = {}
        height, width = cv2.imread(images_path + "/" + name + ".jpg").shape[:2]
        file_name = name + ".jpg"
        image["file_name"] = file_name
        image["height"] = height
        image["width"] = width
        image["id"] = name
        images.append(image)
        for line in f.readlines():
            annotation = {}
            line = line.replace("\n", "")
            if line.endswith(","):  # filter data
                line = line.rstrip(",")
            line_list = [int(i) for i in line.split(",")]
            bbox_xywh = [line_list[0], line_list[1], line_list[2], line_list[3]]
            annotation["image_id"] = name
            annotation["score"] = line_list[4]
            annotation["bbox"] = bbox_xywh
            annotation["category_id"] = int(line_list[5])
            annotation["id"] = id_num
            annotation["iscrowd"] = 0
            annotation["segmentation"] = []
            annotation["area"] = bbox_xywh[2] * bbox_xywh[3]
            id_num += 1
            annotations.append(annotation)
        dataset_dict = {}
        dataset_dict["images"] = images
        dataset_dict["annotations"] = annotations
        dataset_dict["categories"] = categories
        json_str = json.dumps(dataset_dict)
        with open(f'./output.json', 'w') as json_file:
            json_file.write(json_str)
    print("json file write done...")

if __name__ == '__main__':
    test()
