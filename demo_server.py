import argparse
import falcon
import os
import re
from hparams import hparams, hparams_debug_string
from tacotron.demo_synthesizer import Synthesizer
import tensorflow as tf
import time
from tqdm import tqdm
from num_to_text import process_number

html_body = '''<html>
<title>TTS Project</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {
    margin: 0;
    font-family: sans-serif;
    font-size: 14px;
    color: #444
  }

  input {
    font-size: 14px;
    padding: 8px 12px;
    outline: none;
    border: 1px solid #ddd
  }

  input:focus {
    box-shadow: 0 1px 2px rgba(0, 0, 0, .15)
  }

  p {
    padding: 12px
  }

  button {
    background: #28d;
    padding: 9px 14px;
    margin-left: 8px;
    border: none;
    outline: none;
    color: #fff;
    font-size: 14px;
    border-radius: 4px;
    cursor: pointer;
  }

  button:hover {
    box-shadow: 0 1px 2px rgba(0, 0, 0, .15);
    opacity: 0.9;
  }

  button:active {
    background: #29f;
  }

  button[disabled] {
    opacity: 0.4;
    cursor: default
  }

  .topnav {
    overflow: hidden;
    background-color: #333;
  }

  .topnav a {
    float: left;
    display: block;
    color: #f2f2f2;
    text-align: center;
    padding: 14px 16px;
    text-decoration: none;
    font-size: 17px;
  }

  .topnav a:hover {
    background-color: #ddd;
    color: black;
  }

  .active {
    background-color: #4CAF50;
    color: white;
  }

  .topnav .icon {
    display: none;
  }

  .footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: rgb(245, 245, 245);
    color: black;
    font-size: 15px;
    padding: 0;
    text-align: center;
  }

  @media screen and (max-width: 600px) {
    .topnav a:not(:first-child) {
      display: none;
    }
    .topnav a.icon {
      float: right;
      display: block;
    }
  }

  @media screen and (max-width: 600px) {
    .topnav.responsive {
      position: relative;
    }
    .topnav.responsive .icon {
      position: absolute;
      right: 0;
      top: 0;
    }
    .topnav.responsive a {
      float: none;
      display: block;
      text-align: left;
    }
  }
</style>

<body>

  <center>

    <div class="topnav" id="myTopnav">
      <a href="#home" class="active">Home</a>
      <a href="javascript:void(0);" style="font-size:15px;" class="icon" onclick="navResponsive()">&#9776;</a>
    </div>

    <div style="padding-left:16px">
      <h2>Welcome to TTS Project</h2>
      <p>Free project for Vietnamese Text to Speech</p>
    </div>

    <div class="content">
      <img src="https://cdn.dribbble.com/users/410036/screenshots/2236113/sound.gif" alt="Loading" title="Loading" width="30%">
      <form>
        <input id="text" type="text" size="40" placeholder="Enter Text">
        <button id="button" name="synthesize">Speak</button>
      </form>
      <p id="message"></p>
      <audio id="audio" controls autoplay hidden></audio>
    </div>

    <div class="footer">
      <p style="font-size: 16px;">
        <span style="font-size: 22px; color: orange;">&copy;</span> 2018 TTS Project, BK University
      </p>
    </div>

  </center>

  <script type="text/javascript">

    function navResponsive() {
      var x = document.getElementById("myTopnav");
      if (x.className === "topnav") {
        x.className += " responsive";
      } else {
        x.className = "topnav";
      }
    }

    function q(selector) {
      return document.querySelector(selector)
    }

    q('#text').focus()
    q('#button').addEventListener('click', function (e) {
      text = q('#text').value.trim()
      if (text) {
        q('#message').textContent = 'Synthesizing...'
        q('#button').disabled = true
        q('#audio').hidden = true
        synthesize(text)
      }
      e.preventDefault()
      return false
    })
    function synthesize(text) {
      fetch('/synthesize?text=' + encodeURIComponent(text), { cache: 'no-cache' })
        .then(function (res) {
          if (!res.ok) throw Error(response.statusText)
          return res.blob()
        }).then(function (blob) {
          q('#message').textContent = ''
          q('#button').disabled = false
          q('#audio').src = URL.createObjectURL(blob)
          q('#audio').hidden = false
        }).catch(function (err) {
          q('#message').textContent = 'Error: ' + err.message
          q('#button').disabled = false
        })
    }
  </script>

</body>

</html>
'''


class UIResource:
  def on_get(self, req, res):
    res.content_type = 'text/html'
    res.body = html_body


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
    print(text)
    res.data = synthesizer.synthesize(text)
    res.content_type = 'audio/wav'


synthesizer = Synthesizer()
api = falcon.API()
api.add_route('/synthesize', SynthesisResource())
api.add_route('/', UIResource())


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
