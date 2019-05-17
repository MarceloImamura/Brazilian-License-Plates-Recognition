import sys

args = []

args.append('flow')

args.append("--model")
args.append("cfg/yolov2-tiny-voc-1c.cfg")

args.append("--load")
#args.append("bin/yolov2-tiny-voc.weights")
args.append("-1")

args.append("--train")

args.append("--annotation")
args.append("train/annotation")

args.append("--dataset")
args.append("train/ImgPlates")

args.append("--summary")
args.append("summary/")

args.append("--epoch")
args.append("100")

args.append("--batch")
args.append("16")

print("\n\n\n")

for i in args:
    print(i, end = " ")

print("\n\n\n")

from darkflow.cli import cliHandler
cliHandler(args)
