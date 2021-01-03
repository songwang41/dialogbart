
import transformers
import dialogbart
from dialogbart.modeling_dialogbart import LearnedSpeakerEmbeddingV2
from dialogbart import DialogBartTokenizer
tokenizer = DialogBartTokenizer.from_pretrained("/mnt/d/data/data/train_dev_conv_sum/dialogbart/data/distilbart-xsum-12-3-dialog")
roles_map={18497 : 1, 19458:2 }
speaker_embs = LearnedSpeakerEmbeddingV2(10, 3, 1, 1721, roles_map=roles_map)
text = " Issue | Agent : a  | Customer : b b | Agent : c c"
input_ids = tokenizer([text], return_tensors='pt',padding="max_length", max_length = 20)['input_ids']
#   tensor([[    0, 25422,  1721, 18497,  4832,    10,  1437,  1721, 19458,  4832,
#           741,   741,  1721, 18497,  4832,   740,   740,     2,     1,     1]])
speaker_embs.get_speaker_ids(input_ids)
#tensor([[0, 0, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1]])
speaker_embs(input_ids)