import json, argparse
from pprint import pprint
import os, sys

parser = argparse.ArgumentParser(description="coco json format to WIDER Pedestrian submission txt file.")
parser.add_argument('--input', help='path to coco json format file', default=None, type=str)
parser.add_argument('--output', help='path to result submission file', default=None, type=str)
parser.add_argument('--thresh', help='threshold to confidence score', default=0.5, type=float)

if len(sys.argv) < 4:
    parser.print_help() 
    sys.exit(1)

args = parser.parse_args()

JSON_FILE_NAME = args.input # 'bbox_coco_2014_minival_results.json'
THRESH = args.thresh
OUTPUT_FILE_NAME = args.output #'detectron_model_rcnn_1conv_thresh_' + str(THRESH) + "_box_number_"


with open(JSON_FILE_NAME) as f:
    data = json.load(f)

data = filter(lambda x: float(x['score']) >= THRESH, data)

output_dict = [x for x in data if float(x['score']) >= THRESH]

OUTPUT_FILE_NAME += str(len(output_dict))
OUTPUT_FILE_NAME += '.txt'

with open(OUTPUT_FILE_NAME, 'w') as fw:
    for item in output_dict:
        result_string = ""
        image_id = str(item['image_id']).zfill(5)
        image_id = "img" + image_id + ".jpg"
        result_string += image_id
        result_string += " "
        result_string += str(item['score'])
        for bbox in item['bbox']:
            result_string += " "
            result_string += str(bbox)

        fw.write(result_string + "\n")
