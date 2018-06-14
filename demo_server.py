import argparse
import falcon
import os
import re
import numpy as np
from hparams import hparams, hparams_debug_string
from tacotron.demo_synthesizer import Synthesizer
import tensorflow as tf
import time
from tqdm import tqdm
from num_to_text import process_number
from normalization.data_load import load_source_vocab, load_target_vocab

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

def text_normalize(input, batch_size=1, maxlen=35):
  # We use our "load_graph" function
  graph = load_graph('./normalization/infer/infer.pb')

  src2idx, idx2src = load_source_vocab()
  tgt2idx, idx2tgt = load_target_vocab()
  # for op in graph.get_operations():
  #     print(op.name)
  preds = graph.get_tensor_by_name('prefix/ToInt32:0')
  x = graph.get_tensor_by_name('prefix/Placeholder:0')
  y = graph.get_tensor_by_name('prefix/Placeholder_1:0')
  with tf.Session(graph=graph) as sess:
    result = np.zeros((batch_size, maxlen), np.int32)
    input_sent = (input + " . </s>").split()
    feed_x = [src2idx.get(word.lower(), 1) for word in input_sent]
    feed_x = np.expand_dims(np.lib.pad(feed_x, [0, maxlen - len(feed_x)], 'constant'), 0)
    for j in range(maxlen):
        _preds = sess.run(preds, {x: feed_x, y: result})
        result[:, j] = _preds[:, j]
    result = result[0]
    # print('Input  : ', input)
    raw_output = [idx2tgt[idx] for idx in result[result != 3]]

    # Unknown token aligning
    for idx, token in enumerate(feed_x[0]):
        if token == 3:
            break
        if token == 1:
            raw_output[idx] = input_sent[idx]
        if input_sent[idx].istitle():
            raw_output[idx] = raw_output[idx].title()
    # print('Output : ', ' '.join(raw_output[:raw_output.index(".")]))
    return ' '.join(raw_output[:raw_output.index(".")])

class MainPage:
  def on_get(self, req, resp):
    resp.status = falcon.HTTP_200
    resp.content_type = 'text/html'
    with open('statics/index.html', 'r') as f:
      resp.body = f.read()

class AboutPage:
  def on_get(self, req, resp):
    resp.status = falcon.HTTP_200
    resp.content_type = 'text/html'
    with open('statics/about.html', 'r') as f:
      resp.body = f.read()

class SynthesisResource:
  def on_get(self, req, res):
    if not req.params.get('text'):
      raise falcon.HTTPBadRequest()
    string = ",.!?\'-();:\""
    text = req.params.get('text')
    text = process_number(text)
    for char in string:
      text = text.replace(char, ' ')
    text = re.sub("\s\s+", " ", text)
    text = text.strip().lower() + "."
    print('Text: ', text)
    if (req.params.get('accent') == 'true'):
      text = text_normalize(text)
      print('Accent Restoration: ', text)

    res.data = synthesizer.synthesize(text)
    res.content_type = 'audio/wav'


synthesizer = Synthesizer()
api = falcon.API()
api.add_route('/synthesize', SynthesisResource())
api.add_route('/', MainPage())
api.add_route('/about', AboutPage())


if __name__ == '__main__':
  from wsgiref import simple_server
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', \
                      default='logs-Tacotron/pretrained/model.ckpt-258000', \
                      help='Full path to model checkpoint')
  parser.add_argument('--hparams', default='')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)

  synthesizer.load(args.checkpoint)
  print('Serving on port 9000')
  simple_server.make_server('0.0.0.0', 9000, api).serve_forever()
