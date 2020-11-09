import json
import numpy as np
import glob
import cv2
import tqdm
import itertools

def inspect_bbox(ImageDIR):
    print("width__height")
    area = []
    cordinate = []
    width = []
    height = []
    for json_file in glob.glob( ImageDIR + "/*.json"):
        with open(json_file) as f:
            obj = json.load(f)

        annos = obj["regions"]
        
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
            # print("{}__{}".format(w,h))
            width.append(w)
            height.append(h)
            # segs.append(poly)
            bboxes= [minxy[0], minxy[1], w, h]
            area.append(w * h)

    # print(max(np.sqrt(area)))
    # print(min(np.sqrt(area)))
    print("Max width: {}".format(max(width)))
    print("Max height: {}".format(max(height)))
    print("Min width: {}".format(min(width)))
    print("Min height: {}".format(min(height)))
    print("Mean width: {}".format(np.mean(width)))
    print("Mean height: {}".format(np.mean(height)))

def mean_Pixel(ImageDIR):
    allImage = []
    sumPixel = 0
    cOunt = 0
    h_default = 576
    w_default = 768
    for image_file in glob.glob( ImageDIR + "/*.bmp"):
        allImage.append(image_file)
    with tqdm.tqdm(total=len(allImage)) as pbar:
        for idx , image_file in itertools.islice(enumerate(allImage), len(allImage)):
            image = cv2.imread(image_file)
            h, w ,c = image.shape
            # print(image.shape)
            # if h == h_default and w == w_default:
            cOunt += 1
            # print(image_file)
            # print(image.shape)
            meanImage = np.mean(image, axis=tuple(range(image.ndim-1)))
            # print(meanImage)
            sumPixel += meanImage
            pbar.update()

    print("Total mean Pixel: {}".format(sumPixel))
    sumPixel = sumPixel/cOunt
    print("Mean Pixel : {}".format(sumPixel))
    print("Total Images: {}".format(cOunt))
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Inspect dataset.')
    parser.add_argument('--folder', help=""
                "This argument is the path to the input image file", nargs='+')
    parser.add_argument('--image_only', help="",default="1",required=False,
                nargs='+')
    args = parser.parse_args()

    if args.folder:
        # if int(args.image_only) == 0:
        inspect_bbox(args.folder[0])
        mean_Pixel(args.folder[0])