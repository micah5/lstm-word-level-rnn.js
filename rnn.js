require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs');
const fs = require('fs');
var nj = require('numjs');

const maxlen = 30;
const step = 3;
fs.readFile('lyrics_short.txt', 'utf8', function (error, data) {
    if (error) throw error;
    var text = data.toString();
    create_model(text)
});


function onlyUnique(value, index, self) {
  return self.indexOf(value) === index;
}

function indexOfMax(arr) {
    if (arr.length === 0) {
        return -1;
    }
    var max = arr[0];
    var maxIndex = 0;
    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }
    return maxIndex;
}

function sample(preds, temperature) {
  preds = nj.array(preds, 'float64');
  preds = nj.log(preds).divide(temperature)
  exp_preds = nj.exp(preds)
  preds = exp_preds.divide(nj.sum(exp_preds))
  arr = preds.tolist()
  return indexOfMax(arr)
}

async function create_model(text) {
  /* data prep */
  text = text.toLowerCase()
  console.log('corpus length:', text.length)
  var words = text.replace(/(\r\n\t|\n|\r\t)/gm," ").split(" ")
  //console.log("init " + words)
  words = words.filter(onlyUnique)
  //console.log("unique " + words)
  words = words.sort()
  words = words.filter(String)
  //console.log("sorted " + words)

  console.log("total number of unique words" + words.length)

  var word_indices = {}
  var indices_word = {}
  for (let e0 of words.entries()) {
    var idx = e0[0]
    var word = e0[1]
    word_indices[word] = idx
    indices_word[idx] = word
  }
  //console.log("WORD_INDICES" + JSON.stringify(word_indices))
  //console.log("INDICES_WORD" + JSON.stringify(indices_word))

  console.log("maxlen: " + maxlen, " step: " + step)

  var sentences = []
  var sentences1 = []

  var next_words = []
  list_words = text.toLowerCase().replace(/(\r\n\t|\n|\r\t)/gm," ").split(" ").filter(String)
  console.log('list_words ' + list_words.length)

  for (var i = 0; i < (list_words.length - maxlen); i += step) {
    var sentences2 = list_words.slice(i, i + maxlen).join(" ")
    sentences.push(sentences2)
    next_words.push(list_words[i + maxlen])
  }
  console.log('nb sequences(length of sentences):', sentences.length)
  console.log("length of next_word", next_words.length)

  console.log('Vectorization...')
  var X = nj.zeros([sentences.length, maxlen, words.length])
  console.log('X shape' + X.shape)
  var y = nj.zeros([sentences.length, words.length])
  console.log('y shape' + y.shape)
  for (let e of sentences.entries()) {
    var i = e[0]
    var sentence = e[1]
    for (let e2 of sentence.split(" ").entries()) {
      var t = e2[0]
      var word = e2[1]
      X.set(i, t, word_indices[word], 1)
    }
    y.set(i, word_indices[next_words[i]], 1)
  }

  console.log('Creating model... Please wait.');

  console.log("MAXLEN " + maxlen + ", words.length " + words.length)
  var model = tf.sequential();
  model.add(tf.layers.lstm({
    units: 128,
    returnSequences: true,
    inputShape: [maxlen, words.length]
  }));
  model.add(tf.layers.dropout(0.2))
  model.add(tf.layers.lstm({
    units: 128,
    returnSequences: false
  }));
  model.add(tf.layers.dropout(0.2))
  model.add(tf.layers.dense({units: words.length, activation: 'softmax'}));

  model.compile({loss: 'categoricalCrossentropy', optimizer: tf.train.rmsprop(0.002)});

  x_tensor = tf.tensor3d(X.tolist(), null, 'bool')
  //x_tensor.print(true)
  y_tensor = tf.tensor2d(y.tolist(), null, 'bool')
  //y_tensor.print(true)

  /* training */
  await model.fit(x_tensor, y_tensor, {
    epochs: 100,
    batchSize: 32,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(logs.loss + ",")
      }
    }
  })

  /* prediction */
  var start_index = Math.floor(Math.random() * (list_words.length - maxlen - 1))
  var diversity = 0.5
  console.log('----- diversity:', diversity)
  var generated = ''
  var sentence = list_words.slice(start_index, start_index + maxlen + 1)
  generated += sentence.join(" ")
  console.log(generated)
  var str = ""
  for (var i = 0; i < 100; i++) {
    var x_pred = nj.zeros([1, maxlen, words.length])
    var str_b = ""
    for (e3 of sentence.entries()) {
      t = e3[0]
      word = e3[1]
      x_pred.set(0, t, word_indices[word], 1)
      str_b += '(0, ' + t + ", " + word_indices[word] + "), "
    }
    var test = tf.tensor3d(x_pred.tolist())
    //test.print(true)
    const output = model.predict(test)
    //output.print(true)
    var output_data = await output.dataSync()
    var preds = Array.prototype.slice.call(output_data);
    var next_index = sample(preds, diversity)
    var next_word = indices_word[next_index]
    generated += " " + next_word
    sentence.shift()
    str += next_word + " "
  }
  console.log(str)
}
