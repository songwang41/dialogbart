from rouge import Rouge #git clone https://github.com/pltrdy/rouge, python implmentation, slightly different from origianl rouge
from tqdm import tqdm
import pandas as pd
import json
import fire
import pickle
import re

rouge = Rouge()

'''
rouge.get_scores(hypothesis, reference)
[
  {
    "rouge-1": {
      "f": 0.4786324739396596,
      "p": 0.6363636363636364,
      "r": 0.3835616438356164
    },
    "rouge-2": {
      "f": 0.2608695605353498,
      "p": 0.3488372093023256,
      "r": 0.20833333333333334
    },
    "rouge-l": {
      "f": 0.5128205081276938,
      "p": 0.6818181818181818,
      "r": 0.410958904109589
    }
  }
]
'''

def clean_text(text):
    """
    repeat turns: e.g.
    > Customer : pii pii pii pii pii pii pii pii pii pii pii
    Empty turns: e.g.
    > Customer : |
    """
    text = re.sub("( pii)+", " pii", text)
    text = re.sub(" Customer : \|", "", text)
    text = re.sub(" Agent : \|", "", text)
    return text

def truncate_turn(turn_text, max_num_words=100):
    text = ' '.join(turn_text.split()[:max_num_words])
    return text

def clean_transcript(text, turn_sep=" | "):
    text = clean_text(text)
    turns = text.split(turn_sep)
    new_turns = list(map(truncate_turn, turns))
    return turn_sep.join(new_turns)

def calculate_rouge(hyp, ref):
    if "\nSolution : " in ref:
        issue_solution = ref.split("\nSolution : ")
        if len(issue_solution) != 2:
            return 0.0, 0.0
        try: 
            issue = issue_solution[0].replace("Issue : ", "")
            solution = issue_solution[1].replace("<|endoftext|>" ,"")
            hyp = clean_transcript(hyp)
            return rouge.get_scores(hyp.lower(), issue.lower())[0]['rouge-1']['f'], rouge.get_scores(hyp.lower(), solution.lower())[0]['rouge-1']['f']
        except:
            print(hyp)
            print(ref)
            return -1.0, -1.0
    else:
        return 0.0, 0.0


def calculate_rouge_file(input_file, output_file=None):
    '''calculate the rouges and save into a pickle file'''
    with open(input_file) as f_in:
        data = [json.loads(line) for line in f_in]
    scores = []
    for i in tqdm(range(len(data))):
        item = data[i]
        s = calculate_rouge(item['context'], item['completion'])
        scores.append(s)
    
    data_df =pd.DataFrame({"issue_scores": [s[0] for s in scores], "solution_scores": [s[1] for s in scores]})
    data_df['solution'] = [item['completion'].split("\nSolution : ")[1].replace("<|endoftext|>" ,"") for item in data]
    data_df['issue'] = [item['completion'].split("\nSolution : ")[0].replace("Issue : ", "") for item in data]

    if output_file is None:
        output_file = input_file+".pkl" 
    file = open(output_file, 'wb')
    # dump information to the file
    pickle.dump(data_df, file)
    
if __name__ == '__main__':
    #"chat_summary_all/train_data_gpt3_ft/test.json"
    fire.Fire(calculate_rouge_file)