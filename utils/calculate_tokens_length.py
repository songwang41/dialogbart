from transformers import BartTokenizer,  MBartTokenizer
import numpy as np
import fire 



def calculate_avg_token_length(input_file, tokenizer='bart'):
    if tokenizer =='bart':
        #from transformers import BartTokenizer
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        print('Using Bart tokenier facebook/bart-large ....')
    elif tokenizer =='mbart':
        #from transformers import MbartTokenizer
        tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
        print('Using MBart tokenier facebook/mbart-large-cc25....')
    else:
        print("Currently only support bart or mbart!")
    with open(input_file) as f_in:
        lengths = []
        lengths_char = []
        for line in f_in:
            lengths_char.append(len(line.rstrip()))
            l = len(tokenizer.tokenize(line.rstrip()))
            lengths.append(l)
    print(f"n_obs: {len(lengths)}")
    print(f"avg_length :  {round(sum(lengths)/len(lengths), 2)}")
    quantiles = [0.0,0.25,0.5,0.75,0.90,0.95,0.99,1.00]
    print(quantiles)
    print(np.quantile(lengths, quantiles))
    #print(f"max_length: {max(lengths)}")
    #print(f"min_length: {min(lengths)}")
    #print(f"nums less than 11 tokens: {len([x for x in lengths if x<11])}")
    #print(f"nums more than 142 tokens: {len([x for x in lengths if x>142])}")
    #print(f"max_length_char: {max(lengths_char)}")
    print("-------------------- lengths_char distribution -------------")
    print(np.quantile(lengths_char, quantiles))
    print(f"nums of length > 155 chars: {len([x for x in lengths_char if x>155])}")
    return None

if __name__ == '__main__':
  fire.Fire(calculate_avg_token_length)