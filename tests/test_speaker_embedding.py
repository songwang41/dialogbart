
import transformers
import dialogbart
from dialogbart.modeling_dialogbart import LearnedSpeakerEmbeddingV2
from dialogbart import DialogBartTokenizer
tokenizer = DialogBartTokenizer.from_pretrained("/mnt/d/data/data/train_dev_conv_sum/dialogbart/data/distilbart-xsum-12-3-dialog")
speaker_ids = [18497,19458]
speaker_embs = LearnedSpeakerEmbeddingV2(10, 3, 1, 1721, speaker_ids=speaker_ids )

text = " Issue | Agent : a  | Customer : b b | Agent : c c | Other : who are you?"
input_ids = tokenizer([text], return_tensors='pt',padding="max_length", max_length = 30)['input_ids']
input_ids

speaker_ids = speaker_embs.get_speaker_ids(input_ids)
speaker_ids

speaker_embs(input_ids)


turn_token_id = tokenizer.convert_tokens_to_ids('Ġ|')
#1721
padding_idx = tokenizer.convert_tokens_to_ids('<pad>')
#1
tokenizer.convert_tokens_to_ids('ĠCustomer')
#19458
tokenizer.convert_tokens_to_ids('ĠAgent')
#18497
