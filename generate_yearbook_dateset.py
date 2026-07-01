import os
import re
import cv2
import sys
import json
import pickle
import argparse
import numpy as np



def generate_yearbook_dataset():

  runners,json_file = parse_arguments()

  test = []
  train = []

  skipped_videos = 0

  for runner in runners:

    valid_years = sum(1 for year in runners[runner] if len(runners[runner][year]) == 3)

    is_test_runner = valid_years >= 2

    if is_test_runner:
        test.append(runner)
    else:
        train.append(runner)

    for year in runners[runner]:

      is_complete_year = len(runners[runner][year]) == 3

      for recording_point in runners[runner][year]:

        entry = runners[runner][year][recording_point]
        video_name = os.path.basename(entry['mp4'])

        if is_test_runner and not is_complete_year:
          skipped_videos += 1
          print(f"skipping test video {video_name}")
          continue

        print(f"processing {'test' if is_test_runner else 'train'} video {video_name}")
        split_video(entry['mp4'],entry['png'])
        generate_pkl(entry['png'],entry['pkl'])

  print(f"job done, {skipped_videos} skipped videos")

  data = {"TRAIN_SET": train, "TEST_SET": test}
  with open(json_file,"w") as f:
    json.dump(data, f, indent=4)
    f.write("\n")



def split_video(input_path, output_path):
    
    frame_count = 0
    video = cv2.VideoCapture(input_path)

    if not video.isOpened():
      print_error("cannot open video",input_path)
      return

    video_name = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(output_path,exist_ok=True)

    while True:
      ret,frame = video.read()
      if not ret: break
      cv2.imwrite(os.path.join(output_path,f"{video_name}_{frame_count:06d}.png"), frame)
      frame_count += 1

    video.release()



def generate_pkl(input_path, output_path,image_size=64):

  output = []

  for image_file in sorted(os.listdir(input_path)):

    image = cv2.imread(os.path.join(input_path,image_file), cv2.IMREAD_GRAYSCALE)
    if image is None:
      continue

    if image.sum() <= 10000:
      continue

    y_sum = image.sum(axis=1)
    y_top = (y_sum!=0).argmax(axis=0)
    y_bottom = (y_sum!=0).cumsum(axis=0).argmax(axis=0)
    image = image[y_top:y_bottom+1,:]

    if image.shape[0] == 0:
      continue

    ratio = image.shape[1]/image.shape[0]
    image = cv2.resize(image, (int(image_size*ratio),image_size), interpolation=cv2.INTER_CUBIC)

    x_csum = image.sum(axis=0).cumsum()
    x_center = None

    image_sum = image.sum()
    for index,csum in enumerate(x_csum):
      if csum > image_sum/2:
        x_center = index
        break

    if x_center is None:
      continue

    half_width = image_size//2
    left = x_center - half_width
    right = x_center + half_width

    if left <= 0 or right >= image.shape[1]:
      pad = np.zeros((image.shape[0], half_width),dtype=image.dtype)
      image = np.concatenate([pad,image,pad], axis=1)
      left += half_width
      right += half_width

    output.append(image[:,left:right])

  if not output:
    return

  os.makedirs(output_path, exist_ok=True)

  with open(os.path.join(output_path,"0.pkl"), "wb") as f:
    pickle.dump(np.asarray(output), f)



def parse_arguments():

  argument_parser = argparse.ArgumentParser()
  argument_parser.add_argument("input", help="input directory")
  argument_parser.add_argument("output", help="output directory")
  arguments = argument_parser.parse_args()

  runners = {}
  errors_detected = False

  if not os.path.exists(arguments.input):
    errors_detected = print_error("input directory does not exist")

  else:

    pattern = re.compile(r"^(\d{3})_([0-4])_([1-3])\.mp4$")

    for video in sorted(os.listdir(arguments.input)):
      
      video_path = os.path.join(arguments.input,video)
      if not os.path.isfile(video_path):
        errors_detected = print_error("unexpected element in the input directory",video)
        continue

      pattern_match = pattern.match(video)
      if not pattern_match:
        errors_detected = print_error("unexpected file in the input directory",video)
        continue

      runner,year,recording_point = pattern_match.groups()
      entry = runners.setdefault(runner,{}).setdefault(year,{}).setdefault(recording_point,{})
      entry['mp4'] = video_path
      entry['png'] = os.path.join(arguments.output,"png",runner,f"{year}.{recording_point}")
      entry['pkl'] = os.path.join(arguments.output,"pkl",runner,f"{year}.{recording_point}","0")

    if not runners: errors_detected = print_error("no runners detected")

  if os.path.exists(arguments.output): errors_detected = print_error("output directory already exists")

  if errors_detected:
    sys.exit()
  else:
    return runners,os.path.join(arguments.output,"Yearbook.json")



def print_error(message,value=None):
  if value is None: print(f"[ERROR] {message}")
  else:             print(f"[ERROR] {message}: {value}") 
  return True



if __name__ == "__main__": generate_yearbook_dataset()