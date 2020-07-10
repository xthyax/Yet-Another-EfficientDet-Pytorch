import json
import numpy as np
import glob

# TODO: Debug this script
def create_json_coco_Format(ImageDIR, AnnoDIR, id_image, count_annotation, class_names):
    images = []
    annotation = []
    categories =[]
    for i in range (1, len(class_names) + 1):
            sub_category = {"supercategory": "none"}
            sub_category["id"] = i
            sub_category["name"] = class_names[i]
            categories.append(sub_category)
    for json_file in glob.glob( ImageDIR + "/*.json"):
        with open(json_file) as f:
            obj = json.load(f)

        fname = json_file.split("/")[1].replace("bmp.json", "bmp")
        height = obj["height"]
        width = obj["width"]

        # all_data= {"images": images}
        sub_images = {"file_name": fname}
        sub_images["height"] = height
        sub_images["width"] = width
        sub_images["id"] = id_image

        annos = obj["regions"]
        class_file = obj["classId"]
        
        # class_anno = np.empty((0,4),int)

        for anno_position in annos:
            px = annos[anno_position]["List_X"]
            py = annos[anno_position]["List_Y"]
        
            poly = np.stack((px, py), axis=1) + 0.5
            maxxy = poly.max(axis=0)
            minxy = poly.min(axis=0)
            # print(poly.shape)
            segs = [item for sublist in zip(px,py) for item in sublist]

            # boxes.append([minxy[0], minxy[1], maxxy[0], maxxy[1]])
            w , h = maxxy[0] -minxy [0], maxxy[1] - minxy[1]
            # segs.append(poly)
            bboxes= [minxy[0], minxy[1], w, h]
            class_anno = class_names.index(class_file[np.int(anno_position)])

            sub_annotation = {"segmentation": [segs]}
            sub_annotation["area"] = int(w * h)
            sub_annotation["iscrowd"] = 0
            sub_annotation["image_id"] = id_image
            sub_annotation["bbox"] = bboxes
            sub_annotation["category_id"] = class_anno
            sub_annotation["id"] = count_annotation   
            sub_annotation["ignore"] = 0
            annotation.append(sub_annotation)
            count_annotation += 1
        id_image +=1 
        images.append(sub_images)                
    all_data= {"images": images, "annotations": annotation, "categories" : categories}
    # print(all_data)
    with open(AnnoDIR +'/instances_'+ImageDIR+'.json', 'w') as outfile:
        print('Writing JSON file from '+ ImageDIR+ "into" + AnnoDIR +'/instances_'+ImageDIR+'.json')
        json.dump(all_data, outfile)
    """
        Return count_annotation to continue from the last image in Training set
    """
    return id_image, count_annotation
    

def load(TrainImageDIR, ValImageDIR, TestImageDir, AnnoDIR, listClass):
    class_names = listClass
    count_annotation = 1
    id_image = 1

    id_image, count_annotation = create_json_coco_Format(TrainImageDIR, AnnoDIR, id_image, count_annotation, class_names)
    id_image, count_annotation = create_json_coco_Format(ValImageDIR, AnnoDIR, id_image, count_annotation, class_names)
    create_json_coco_Format(TestImageDir, AnnoDIR, id_image, count_annotation, class_names)


class_list = ["BG", "Bridging defect", "Bridging defect 1", "Overkill"]

load('Train_filtered','Validation','Test','annotations', class_list)