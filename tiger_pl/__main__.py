import argparse

from tiger_pl import Tiger

parser = argparse.ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()
print(Tiger(args.filename).execute())
