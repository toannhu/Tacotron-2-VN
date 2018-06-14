import argparse
import os
import numpy as np
import tensorflow as tf
from normalization.data_load import load_source_vocab, load_target_vocab

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="./normalization/infer/infer.pb", type=str,
                        help="Frozen model file to import")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--input", default='an gi bay gio', type=str)
    parser.add_argument("--maxlen", default=35, type=int)
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)

    src2idx, idx2src = load_source_vocab()
    tgt2idx, idx2tgt = load_target_vocab()
    # for op in graph.get_operations():
    #     print(op.name)
    preds = graph.get_tensor_by_name('prefix/ToInt32:0')
    x = graph.get_tensor_by_name('prefix/Placeholder:0')
    y = graph.get_tensor_by_name('prefix/Placeholder_1:0')
    with tf.Session(graph=graph) as sess:
        result = np.zeros((args.batch_size, args.maxlen), np.int32)
        input_sent = (args.input + " . </s>").split()
        feed_x = [src2idx.get(word.lower(), 1) for word in input_sent]
        feed_x = np.expand_dims(np.lib.pad(feed_x, [0, args.maxlen - len(feed_x)], 'constant'), 0)
        for j in range(args.maxlen):
            _preds = sess.run(preds, {x: feed_x, y: result})
            result[:, j] = _preds[:, j]
        result = result[0]
        print('Input  : ', args.input)
        raw_output = [idx2tgt[idx] for idx in result[result != 3]]

        # Unknown token aligning
        for idx, token in enumerate(feed_x[0]):
            if token == 3:
                break
            if token == 1:
                raw_output[idx] = input_sent[idx]
            if input_sent[idx].istitle():
                raw_output[idx] = raw_output[idx].title()
        print('Output : ', ' '.join(raw_output[:raw_output.index(".")]))
