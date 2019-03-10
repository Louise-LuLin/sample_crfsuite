__author__ = 'lulin'

import pickle

prefix = '../dataset/'
source = 'train'

labels = []
strings = []

with open(prefix + source + '_label.txt', 'r') as fp:
	line = 'initial'
	while line:
		line = fp.readline().replace("\n",'')
		label = line.split(',')
		if(len(label) > 1):
			labels.append(label)
with open(prefix + source + '_string.txt', 'r') as fp:
	line = 'initial'
	while line:
		line = fp.readline()
		string = [ch for ch in line[:-1]]
		if(len(string) > 0):
			strings.append(string)
print (len(labels))
print (len(strings))

with open(prefix + source + "_label.bin", "wb") as sod_dataset:
    pickle.dump(labels, sod_dataset)
with open(prefix + source + "_string.bin", "wb") as sod_string:
    pickle.dump(strings, sod_string)