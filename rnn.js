require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const fs = require('fs')
var nj = require('numjs')

const maxlen = 30
const step = 3

/**
 *  Returns index of max in arr
 */
function indexOfMax(arr) {
    if (arr.length === 0) {
        return -1
    }
    let max = arr[0]
    let maxIndex = 0
    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i
            max = arr[i]
        }
    }
    return maxIndex
}

/**
 *  Random dist
 */
function sample(preds, temperature) {
  preds = nj.array(preds, 'float64')
  preds = nj.log(preds).divide(temperature)
  exp_preds = nj.exp(preds)
  preds = exp_preds.divide(nj.sum(exp_preds))
  arr = preds.tolist()
  return indexOfMax(arr)
}

/**
 *  Creates, trains and runs predictions on model
 *  TODO: Split
 */
async function create_model(text) {
  /* data prep */
  text = text.toLowerCase()
  console.log('corpus length:', text.length)

  let words = text.replace(/(\r\n\t|\n|\r\t)/gm," ").split(" ")
  words = words.filter((value, index, self) => self.indexOf(value) === index)
  words = words.sort()
  words = words.filter(String)

  console.log("total number of unique words", words.length)

  const word_indices = {}
  const indices_word = {}
  for (let e0 of words.entries()) {
    const idx = e0[0]
    const word = e0[1]
    word_indices[word] = idx
    indices_word[idx] = word
  }

  console.log("maxlen: " + maxlen, " step: " + step)

  const sentences = []
  const sentences1 = []

  const next_words = []
  let list_words = text.toLowerCase().replace(/(\r\n\t|\n|\r\t)/gm, " ").split(" ")
  list_words = list_words.filter(String)
  console.log('list_words ' + list_words.length)

  for (var i = 0; i < (list_words.length - maxlen); i += step) {
    var sentences2 = list_words.slice(i, i + maxlen).join(" ")
    sentences.push(sentences2)
    next_words.push(list_words[i + maxlen])
  }
  console.log('nb sequences(length of sentences):', sentences.length)
  console.log('length of next_word', next_words.length)

  console.log('Vectorization...')
  const X = nj.zeros([sentences.length, maxlen, words.length])
  console.log('X shape' + X.shape)
  const y = nj.zeros([sentences.length, words.length])
  console.log('y shape' + y.shape)
  for (let e of sentences.entries()) {
    const i = e[0]
    const sentence = e[1]
    for (let e2 of sentence.split(" ").entries()) {
      const t = e2[0]
      const word = e2[1]
      X.set(i, t, word_indices[word], 1)
    }
    y.set(i, word_indices[next_words[i]], 1)
  }

  console.log('Creating model... Please wait.')

  console.log("MAXLEN " + maxlen + ", words.length " + words.length)
  const model = tf.sequential()
  model.add(tf.layers.lstm({
    units: 128,
    returnSequences: true,
    inputShape: [maxlen, words.length]
  }))
  model.add(tf.layers.dropout(0.2))
  model.add(tf.layers.lstm({
    units: 128,
    returnSequences: false
  }))
  model.add(tf.layers.dropout(0.2))
  model.add(tf.layers.dense({units: words.length, activation: 'softmax'}))

  model.compile({loss: 'categoricalCrossentropy', optimizer: tf.train.rmsprop(0.002)})

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
  const start_index = Math.floor(Math.random() * (list_words.length - maxlen - 1))
  const diversity = 0.5
  console.log('----- diversity:', diversity)
  let generated = ''
  const sentence = list_words.slice(start_index, start_index + maxlen + 1)
  generated += sentence.join(" ")
  console.log(generated)
  let str = ""
  for (let i = 0; i < 100; i++) {
    let x_pred = nj.zeros([1, maxlen, words.length])
    let str_b = ""
    for (e3 of sentence.entries()) {
      t = e3[0]
      word = e3[1]
      x_pred.set(0, t, word_indices[word], 1)
      str_b += '(0, ' + t + ", " + word_indices[word] + "), "
    }
    const test = tf.tensor3d(x_pred.tolist())
    const output = model.predict(test)
    const output_data = await output.dataSync()
    const preds = Array.prototype.slice.call(output_data)
    const next_index = sample(preds, diversity)
    const next_word = indices_word[next_index]
    generated += " " + next_word
    sentence.shift()
    str += next_word + " "
  }
  console.log(str)
}

fs.readFile('lyrics_short.txt', 'utf8', (error, data) => {
    if (error) throw error
    create_model(data.toString())
})
