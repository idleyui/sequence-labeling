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

```python
from run import CWS

train_file = 'train.txt'
model_file = 'model.pkl'
result_file = 'result.txt'
test_file = 'test.txt'

if __name__ == '__main__':
    c = CWS('crf')
    c.train(train_file, model_file)
    c.segment(model_file, test_file, result_file)
    c.evaluate(test_file, result_file)
```

## Models

### HMM

