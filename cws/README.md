# Chinese Word Segment(CWS)

## Structure

```bash
.
├── README.md           
├── crf_model.py        CRF Model
├── dataset.py          data preprocess
├── evaluate.py         result evaluate
├── hmm_model.py        HMM Model
└── run.py              running script
```

## Getting Started

```bash
# train
python run.py \
    --method [crf|hmm|lstm-crf] \
    --mode train \
    --train_file [file path] \
    --model_file [file path] \
    
# segment
python run.py \
    --method [crf|hmm|lstm-crf] \
    --mode segment \
    --test_file [file path] \
    --model_file [file path] \
    --result_file [file path]
    
# eval
python run.py \
    --method [crf|hmm|lstm-crf] \
    --mode eval \
    --test_file [file path] \
    --result_file [file path]
    
# e.g. training with hmm method
python cws\run.py --method hmm --mode train --train_file train_cws.txt --model_file cws_hmm_model.pkl
```