# DialogBart: BART (seq2seq) struncture adapted for conversation data

This is currently under development.

The package will be used for conversational text modeling, including language modeling, text generation.


# Developments

## add speaker embedding
option 1: use speaker position

option 2: provide the fixed-role and thus their embedding e.g. " Agent" and " Customer" by setting:
speaker_ids= [18497, 19458]

## add turn position embedding