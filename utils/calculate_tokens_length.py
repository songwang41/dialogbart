from transformers import BartTokenizer
import fire 

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

def calculate_avg_token_length(input_file):
    with open(input_file) as f_in:
        lengths = []
        lengths_char = []
        for line in f_in:
            lengths_char.append(len(line.rstrip()))
            l = len(tokenizer.tokenize(line.rstrip()))
            lengths.append(l)
    print(f"n_obs: {len(lengths)}")
    print(f"avg_length :  {round(sum(lengths)/len(lengths), 2)}")
    print(f"max_length: {max(lengths)}")
    print(f"min_length: {min(lengths)}")
    print(f"nums less than 11 tokens: {len([x for x in lengths if x<11])}")
    print(f"nums more than 142 tokens: {len([x for x in lengths if x>142])}")
    print(f"max_length_char: {max(lengths_char)}")
    print(f"nums of length > 155 chars: {len([x for x in lengths_char if x>155])}")
    print(lengths[:20])
    return None

if __name__ == '__main__':
  fire.Fire(calculate_avg_token_length)