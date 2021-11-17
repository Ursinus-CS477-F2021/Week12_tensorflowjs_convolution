// https://www.tensorflow.org/js/guide/models_and_layers
// https://www.tensorflow.org/js/guide/tensors_operations
// https://www.tensorflow.org/js/guide/train_models


const shape = [2, 2];

const xs = tf.randomNormal([100, 2]);

xs.array().then(x => {
  let ys = [];
  for (let i = 0; i < x.length; i++) {
    let d = Math.sqrt(x[i][0]*x[i][0] + x[i][1]*x[i][1]);
    if (d < 1) {
      ys.push(1);
    }
    else {
      ys.push(0);
    }
  }
  ys = tf.tensor(ys);

  // Create an arbitrary graph of layers, by connecting them
  // via the apply() method.
  const input = tf.input({shape: [2]});
  const dense1 = tf.layers.dense({units: 3, activation: 'relu'}).apply(input);
  const dense2 = tf.layers.dense({units: 1, activation: 'relu'}).apply(dense1);
  const model = tf.model({inputs: input, outputs: dense2});

  model.compile({loss: 'meanSquaredError',
                optimizer: 'sgd'});

  //let ys = [0, 1];


  // Train the model using the data.
  model.fit(xs, ys, {epochs: 10}).then(() => {
    // Use the model to do inference on a data point the model hasn't seen before:
    const test = tf.tensor([0, 0]);
    test.reshape([2]);
    model.predict(test).print();
    // Open the browser devtools to see the output
  });


});

