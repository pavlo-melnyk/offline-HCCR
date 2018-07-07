# Offline Handwritten Chinese Character Classifier
## Description:
- the main part of Master's thesis project;

- recognizes isolated handwritten Chinese characters among 3755 classes of GB2312-80 level-1 standard;

- based on a CNN with the following architecture:

Input - 64C3 - 64C3 - MP2 - 96C3 - 96C3 - MP2 - 128C3 - 128C3 - MP2 - 256C3(Drop) - 256C3(Drop) - MP2 - 512C3(Drop) - 512C3(Drop) - MP2 -
512C3 - GAP(Drop) - 3755Output

 - for more details, you can use the 'architecture.prototxt' file and Netscope CNN Analyzer (by ethereon; extended for CNN Analysis by dgschwend) - http://dgschwend.github.io/netscope/quickstart.html;

- Trained on:
CASIA-HWDB1.0-1.1 datasets collected by National Laboratory of Pattern Recognition (NLPR), Institute of Automation of Chinese Academy of Sciences (CASIA), written by 420 and 300 persons. The overall training dataset contains 2,678,424 samples.

- Evaluated on:
the most common benchmark dataset – ICDAR-2013 competition dataset, containing 224,419 samples written by 60 persons.

- Final accuracy - 97,00%.

- Total number of parameters - 9,145,643 (~35 MB of storage).


## Usage:
 - run 'main.py';
 - follow the prompt instructions:
    - specify the isolated character image/ filepath (e.g. 'imgs/1.png');
    - specify the n number of candidates for recognition - will show n most confident predictions;
    - type 'y' if you want to continue.
    

## Demo:
