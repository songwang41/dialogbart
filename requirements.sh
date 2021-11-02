# if it is cpu
#pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
#pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
#note that torch 1.6.0 will easily give out or memory errors
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch_lightning==1.1.0
pip install transformers==4.1.0   
pip install gitpython==3.1.11 rouge-score==0.0.4 sacrebleu==1.4.13
pip install sentencepiece 
pip install dialogbart

#pip install GitPython
#pip install rouge_score sacrebleu
#pip install sentencepiece requires for some higher version of transformers

