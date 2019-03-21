#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse
import numpy as np
from calibrate_scores import LinearModel

if __name__ == "__main__":
  parser = argparse.ArgumentParser("Apply calibration model to LLR scores")
  parser.add_argument('model')
  parser.add_argument('input_score_file', nargs='+', help="One or more input score files. Each line is a triple <enrolled_speaker> <test_speaker> <LLR_score>") 
  parser.add_argument('output_score_file')

  args = parser.parse_args()
  
  model = LinearModel(len(args.input_score_file))
  model.load_state_dict(torch.load(args.model))
  model.eval()
  model.double()
  
  input_tensor = None
  for input_score_file in args.input_score_file:
    input_keys_and_scores = []
    for l in open(input_score_file):
      ss = l.split()
      input_keys_and_scores.append((ss[0] + " " + ss[1], float(ss[2])))

    if input_tensor is None:
      input_tensor = torch.tensor([i[1] for i in input_keys_and_scores], dtype=torch.float64).reshape(-1, 1)
    else:
      input_tensor = torch.cat((input_tensor, torch.tensor([i[1] for i in input_keys_and_scores], dtype=torch.float64).reshape(-1, 1)), dim=1)
  
  output_tensor = model(input_tensor)

  with open(args.output_score_file, "w") as f_out:
    for i, s in enumerate(input_keys_and_scores):
      print(s[0], output_tensor[i].item(), file=f_out)