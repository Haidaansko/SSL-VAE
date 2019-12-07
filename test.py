import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='M2')
args = parser.parse_args()
print(args.model)