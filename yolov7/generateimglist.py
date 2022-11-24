import os
pathdir = '/opt/ml/dataset/test'
files_list = os.listdir(pathdir)

listfile = open("test.txt", 'w')
for file_list in files_list:
    if file_list[0] != '.':
        listfile.write(pathdir + "/" + file_list + "\n")
listfile.close()
