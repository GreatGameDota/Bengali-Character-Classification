import subprocess

import json
import zipfile
import os
os.chdir('data/')

subprocess.run(["pip", "install", "kaggle", "--quiet"])

subprocess.run(["pip", "install", "pytorchcv", "--quiet"])
subprocess.run(["pip", "install", "torchtoolbox", "--quiet"])
subprocess.run(["pip", "install", "iterative-stratification", "--quiet"])

if not os.path.isfile('test.csv'):
  subprocess.run(["kaggle", "competitions", "download", "-c", "bengaliai-cv19"])
  subprocess.run(["Expand-Archive", "-Force", "data/bengaliai-cv19.zip", "data/"])

if not os.path.isfile('train/Train_0.png'):
  subprocess.run(["kaggle", "datasets", "download", "iafoss/grapheme-imgs-128x128"])
  subprocess.run(["Expand-Archive", "-Force", "data/grapheme-imgs-128x128.zip", "data/train/"])
