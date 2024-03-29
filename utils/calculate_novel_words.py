import fire 
from collections import Counter, defaultdict
import nltk
#nltk_tokens = nltk.word_tokenize(word_data)

class NovelWords:
    def __init__(self, source_file, target_file, output_tsv_file=None, 
    sep_or_tokenizer=" ", jargon_words_file=None):
        with open(source_file) as f_in:
            self.src_texts = [line.strip() for line in f_in]
        with open(target_file) as f_in:
            self.tgt_texts = [line.strip() for line in f_in]
        if jargon_words_file:
            with open(jargon_words_file) as f_in:
                self.jargon_words=[line.strip() for line in f_in]
        else:
            self.jargon_words = []
        assert len(self.src_texts) == len(self.tgt_texts)
        #record the frequency of novel words
        self.novel_tokens=defaultdict(int)
        #record the ratio of novel words in each <src,tgt> pair
        self.novel_tokens_ratio=[]
        if output_tsv_file is None:
            self.output_tsv_file="novel_tokens_frequency.tsv"
        else:
            self.output_tsv_file=output_tsv_file
        # seprator or tokenizer to generate tokens
        self.sep_or_tokenizer=sep_or_tokenizer
        

    def tokenize(self, text):
        if self.sep_or_tokenizer == " " or self.sep_or_tokenizer=='space':
            return text.split()
        elif self.sep_or_tokenizer == "nltk":
            return  nltk.word_tokenize(text)
        else:
            assert False, "sep_or_tokenizer should be in ' ' or 'nltk'"

    def process_one_pair(self, src_text, tgt_text):

        tgt_tokens = self.tokenize(tgt_text)
        src_tokens = self.tokenize(src_text)
        tgt_tokens_count = Counter(tgt_tokens)
        src_tokens_count = Counter(src_tokens)
        new_tokens_count = 0
        if self.jargon_words:
            with open("target_text_novels_restricted.txt", 'a+') as f_out:
                for key in self.jargon_words:
                    if key in tgt_tokens_count and not key in src_tokens_count:
                        f_out.write(key+'\t')
                        self.novel_tokens[key] += 1 #tgt_tokens_count[key]
                        new_tokens_count += tgt_tokens_count[key]
                self.novel_tokens_ratio.append(new_tokens_count/len(tgt_tokens))
                f_out.write('\n')
        else:
            with open("target_text_novels.txt", 'a+') as f_out: 
                for key, value in tgt_tokens_count.items():
                    if not key in src_tokens_count:
                        f_out.write(key+'\t')
                        self.novel_tokens[key] += 1 
                        new_tokens_count += tgt_tokens_count[key]
                self.novel_tokens_ratio.append(new_tokens_count/len(tgt_tokens))
                f_out.write('\n')
            
    def aggregate_metric_novel_tokens(self):
        for src, tgt in zip(self.src_texts, self.tgt_texts):
            self.process_one_pair(src, tgt)
    
    def print_metrics(self):
        print(f"number of pairs: {len(self.tgt_texts)}")
        print(f"mean novel_tokens_ratio: is {sum(self.novel_tokens_ratio)/len(self.novel_tokens_ratio)}")
        print(f"% summaries contain novel words:{round(len([x for x in self.novel_tokens_ratio if x>1e-6])/len(self.novel_tokens_ratio)*100, 2)}")
        # return the list of keys whose values are sorted in order.
        sorted_keys = sorted(self.novel_tokens, key=self.novel_tokens.get, reverse=True) 
        with open(self.output_tsv_file, 'w') as f_out:
            f_out.write("tokens\tfrequency\n")
            for k in sorted_keys:
                f_out.write(f"{k}\t{self.novel_tokens[k]}\n")
        with open("agg_metrics.txt", 'w') as f_out:
            f_out.write(f"n_obs:\n{len(self.tgt_texts)}\n")
            f_out.write(f"% novel words are novel in summary:\n{round(sum(self.novel_tokens_ratio)/len(self.novel_tokens_ratio)*100, 2)}\n")
            f_out.write(f"% summaries contain novel words:\n{round(len([x for x in self.novel_tokens_ratio if x>1e-6])/len(self.novel_tokens_ratio)*100, 2)}\n")
        
        with open("agg_metrics.txt", 'w') as f_out:
            f_out.write(f"n_obs:\n{len(self.tgt_texts)}\n")
            f_out.write(f"% novel words are novel in summary:\n{round(sum(self.novel_tokens_ratio)/len(self.novel_tokens_ratio)*100, 2)}\n")
            f_out.write(f"% summaries contain novel words:\n{round(len([x for x in self.novel_tokens_ratio if x>1e-6])/len(self.novel_tokens_ratio)*100, 2)}\n")
        print("novel_tokens frequency is written to novel_tokens_frequency.tsv ")

def calculate_novel_words_metrics(source_file, 
                                target_file, 
                                output_tsv_file=None, 
                                sep_or_tokenizer=" ",
                                jargon_words_file=None):
    #src_file = "data/research/samsum/samsum_hf/test.source"
    #tgt_file = "data/research/samsum/dialogbart_large_default_cnn_batch64_6epochs_min11-lenpen1-10ep/test_generations.txt"
    novelwords = NovelWords(source_file, target_file, output_tsv_file, sep_or_tokenizer, jargon_words_file)
    novelwords.aggregate_metric_novel_tokens()
    novelwords.print_metrics()
    
if __name__ == '__main__':
    fire.Fire(calculate_novel_words_metrics)
    

#example command
'''
python calculate_novel_words.py \
--source_file ../scripts/data/train_dev_conv_sum/chat_summary_combined_xbox/train_data_v2/val.source \
--target_file ../scripts/data/train_dev_conv_sum/chat_summary_combined_xbox/train_data_v2/val.target \
--output_tsv_file val_novel_words_frequency_nltk.tsv \
--sep_or_tokenizer nltk
'''
#with jargon words
'''
python calculate_novel_words.py \
--source_file ../scripts/data/train_dev_conv_sum/chat_summary_combined_xbox/train_data_v2/val.source \
--target_file ../scripts/data/train_dev_conv_sum/chat_summary_combined_xbox/train_data_v2/val.target \
--output_tsv_file val_novel_words_frequency_nltk_restricted.tsv \
--sep_or_tokenizer nltk \
--jargon_words_file jargon_words_nltk.txt
'''