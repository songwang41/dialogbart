
# Test 1: loading original bart models
# Some weights of DialogBartForConditionalGeneration were not initialized from the model checkpoint at facebook/bart-large 
# and are newly initialized: ['encoder.embed_turns.weight', 'encoder.embed_speakers.weight']

from dialogbart import DialogBartTokenizer, DialogBartForConditionalGeneration
model = DialogBartForConditionalGeneration.from_pretrained("facebook/bart-large", force_bos_token_to_be_generated=True)
tok = DialogBartTokenizer.from_pretrained("facebook/bart-large")
example_english_phrase = "UN Chief Says There Is No <mask> in Syria"
batch = tok(example_english_phrase, return_tensors='pt')
generated_ids = model.generate(batch['input_ids'])
#dialogbart
assert tok.batch_decode(generated_ids, skip_special_tokens=True) == ['UN Chief Says There Is No Plan to Stop ISIS in Syria']



if False:
#bart model
from transformers import BartForConditionalGeneration, BartTokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", force_bos_token_to_be_generated=True)
tok = BartTokenizer.from_pretrained("facebook/bart-large")
example_english_phrase = "UN Chief Says There Is No <mask> in Syria"
batch = tok(example_english_phrase, return_tensors='pt')
generated_ids = model.generate(batch['input_ids'])
assert tok.batch_decode(generated_ids, skip_special_tokens=True) == ['UN Chief Says There Is No Plan to Stop Chemical Weapons in Syria']


# finetuned dialog pretrained model

from dialogbart import DialogBartTokenizer, DialogBartForConditionalGeneration
model_name_or_path="/home/sonwang/work/train_dev_conv_sum/chat_summary_v4/checkpoints/best_tfmr"
model = DialogBartForConditionalGeneration.from_pretrained(model_name_or_path, force_bos_token_to_be_generated=True)
tok = DialogBartTokenizer.from_pretrained(model_name_or_path)
example_english_phrase = "UN Chief Says There Is No <mask> in Syria"
batch = tok(example_english_phrase, return_tensors='pt')
generated_ids = model.generate(batch['input_ids'])
#dialogbart
assert tok.batch_decode(generated_ids, skip_special_tokens=True) == ['UN Chief Says There Is No Plan to Stop ISIS in Syria']
