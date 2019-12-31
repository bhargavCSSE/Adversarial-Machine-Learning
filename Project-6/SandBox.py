# import random
# chromosome_length = 15
# chromosome = []
# for i in range(chromosome_length):
#     chromosome.append(random.choice([0, random.randint(100, 1000)]))
#
# for i in range(len(chromosome)):
#     if chromosome[i]:
#         print(chromosome[i])

# import pandas as pd
# import numpy as np
# df_X = pd.read_excel('All.xlsx', sheet_name='All')
# df_Y = pd.read_excel('All.xlsx', Sheet_name='Out')
# X_train = np.array(df_X)
# Y_train = np.array(df_Y)
#
# print(Y_train[0])
# import pandas as pd
# runCount = 10
# new_chromosome = []
# df = pd.DataFrame(columns=["Run", "HiddenLayer", "Accuracy"])
# for i in range(runCount):
#
#     for j in range(len(chromosome)):
#         if chromosome[j]:
#             new_chromosome.append(chromosome[j])
#
#     df = df.append(dict(Run=i,
#                    HiddenLayer=new_chromosome,
#                    Accuracy=0.01), ignore_index=True)
#     new_chromosome=[]
#
# print(df)