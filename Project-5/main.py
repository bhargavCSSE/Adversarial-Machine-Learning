import Data_Utils
import pandas as pd
from Extractor.DatasetInfo import DatasetInfo
from Extractor.Extractors import BagOfWords, Stylomerty, Unigram, CharacterGram

data_dir = "./data/"
feature_set_dir = "./datasets/"

if __name__ == "__main__":
    for i in range(4):
        if i == 0:
            extractor = Unigram(data_dir + "CASIS25/", "casis25")
        elif i == 1:
            extractor = Stylomerty(data_dir + "CASIS25/", "casis25")
        elif i == 2:
            extractor = BagOfWords(data_dir + "CASIS25/", "casis25")
        else:
            extractor = CharacterGram(data_dir + "CASIS25/", "casis25", gram=3, limit=1000)

        extractor.start()
        lookup_table = extractor.lookup_table
        print("Generated Lookup Table:")
        # print(lookup_table)
        col = []
        if lookup_table is not False:
            print("'" + "', '".join([str("".join(x)).replace("\n", " ") for x in lookup_table]) + "'")
            for x in lookup_table:
                col.append("'" + "', '".join([str("".join(x)).replace("\n", " ")]) + "'")
            generated_file = feature_set_dir + extractor.out_file + ".txt"
            generated_csv_file = feature_set_dir + extractor.out_file + ".csv"
            data, labels = Data_Utils.get_dataset(generated_file)
            df = pd.DataFrame(data, columns=col)
            df.insert(0, "Label", labels, True)
            df.to_csv(generated_csv_file)
        else:
            generated_file = feature_set_dir + extractor.out_file + ".txt"
            generated_csv_file = feature_set_dir + extractor.out_file + ".csv"
            data, labels = Data_Utils.get_dataset(generated_file)
            df = pd.DataFrame(data)
            df.insert(0, "Label", labels, True)
            df.to_csv(generated_csv_file)

        # Get dataset information
        dataset_info = DatasetInfo("casis25_bow")
        dataset_info.read()
        authors = dataset_info.authors
        writing_samples = dataset_info.instances
        print("\n\nAuthors in the dataset:")
        print(authors)

        print("\n\nWriting samples of an author 1000")
        print(authors["1000"])

        print("\n\nAll writing samples in the dataset")
        print(writing_samples)

        print("\n\nThe author of the writing sample 1000_1")
        print(writing_samples["1000_1"])

        # print(labels[0], data[0])
    print("Done")
