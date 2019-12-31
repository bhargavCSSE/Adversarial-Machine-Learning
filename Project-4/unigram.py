import zipfile
import string
import pandas as pd

my_zip = zipfile.ZipFile("CASIS-25_Dataset.zip")
all_files = my_zip.namelist()
all_files = sorted(all_files)
look_up_table = sorted([i for i in string.printable])
look_up_table = look_up_table[5:]

feature_df = pd.DataFrame(columns=look_up_table)

for file in all_files:
    feature_df.loc[file] = 0
    f=my_zip.open(file)
    for line in f:
        for ch in line:
            char = chr(ch)
            if char in look_up_table:
                feature_df.loc[file, char] += 1