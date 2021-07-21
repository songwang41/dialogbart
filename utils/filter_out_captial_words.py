import pandas as pd
import string
import re
import fire
def is_upper(text):
    #there is some float numbers
    text = str(text)
    return text == text.upper()

def is_punctuation(text):
    punc_marks = string.punctuation #punc_marks = '''!"#$%&'()*+, -./:;<=>?@[\]^_`{|}~'''
    for p in punc_marks: 
        text = text.replace(p, "")
    return text
    # text = re.sub(r'[^\w\s]', '', text)
    return text

def filter_data(input_file, output_file):
    #"val_novel_words_frequency_space.tsv"
    #"val_novel_words_frequency_space_upper.tsv"
    data = pd.read_csv(input_file, 
        dtype = {'tokens':str, 'frequency':int}, 
        sep="\t")
    print(len(data))
    print(data[:5])
    data_upper = data[data['tokens'].apply(lambda x: is_upper(x))]
    print(len(data_upper))
    data_upper.to_csv(output_file, index=False, sep="\t")

if __name__=='__main__':
    fire.Fire(filter_data)