# Vietnamese Text Accent Restoration

## Requirements
* numpy  >= 1.11.1
* TensorFlow >= 1.2

## Transformer Model
The Transformer is a novel approach which was proposed in the paper Attention is All You Need [arxiv](https://arxiv.org/abs/1706.03762). The author claimed that
their model outperformer SOTA works using only the attention mechanism. Neat.

## Accent Restoration for Vietnamese
Vietnamese are still occasionally written without tones and diacritical marks, for example: "co gai dam dang", which maybe interpreted as "cô gái đảm đang" or, well, you know...

The Transformer is effectively a seq2seq model and thus can be apply directly to this problem. The input is a vector of accent-free syllables, for example:

\['toi', 'di', 'choi', 'cong','vien'\]

And the output is the actual, expected syllable vector:

\['tôi', 'đi', 'chơi', 'công','viên'\]

## Results

The bigger model achieved 97.5% per-syllable accuracy and the smaller one 93.4%.


## Demo

The smaller model was packed with this repo. Run the following command:

    python demo.py --input <your-raw-sentence>

For example:

    python demo.py --input 'toi la fan cua Manchester United'
Output:

    tôi là fan của Manchester United