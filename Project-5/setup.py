import os
import glob
from shutil import copyfile

TestFileList = "AdversarialTests.txt"
TestPath = "data/CASIS25/"
TestFilePath = 'data/advTest/'

for filename in glob.glob(TestPath + "adv*"):
    os.remove(filename)

f = open(TestFileList,"r")
if f.mode == 'r':
    contents = f.read()
    list = contents.splitlines()
    print(list)
    for filename in list:
        copyfile(TestFilePath + filename, TestPath + filename)