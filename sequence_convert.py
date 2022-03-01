import sys

path = sys.argv[1]

with open(path, 'r') as f:
	data = f.readlines()
	
data1 = data.copy()
data2 = data.copy()

for i, s in enumerate(data):
	x = data[i].split()
	data1[i] = x[0].replace('_','') + '\n'
	data1[i] = data1[i].replace('20210719','')
	data2[i] = x[1].replace('jpg','png') + '\n'
	
#t1 = open('times.txt','w')

t2 = open('images.txt','w')

#for s in data1:
#	t1.write(s)

for s in data2:
	t2.write(s)

#t1.close()
t2.close()

