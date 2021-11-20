// https://www.tensorflow.org/js/guide/models_and_layers
// https://www.tensorflow.org/js/guide/tensors_operations
// https://www.tensorflow.org/js/guide/train_models


const xs = tf.randomNormal([1000, 2]);

xs.array().then(x => {
  let ys = [];
  for (let i = 0; i < x.length; i++) {
    let d = Math.sqrt(x[i][0]*x[i][0] + x[i][1]*x[i][1]);
    if (d < 0.5) {
      ys.push(0); // Label as 0 if it's within a radius of 0.5
    }
    else {
      ys.push(1);
    }
  }
  let ysorig = ys;
  ys = tf.tensor(ys);

  // Create an arbitrary graph of layers, by connecting them
  // via the apply() method.
  const input = tf.input({shape: [2]});
  const dense1 = tf.layers.dense({units: 3, activation: 'sigmoid', useBias:true}).apply(input);
  const dense2 = tf.layers.dense({units: 1, activation: 'sigmoid', useBias:true}).apply(dense1);
  const model = tf.model({inputs: input, outputs: dense2});
  //tfvis.show.modelSummary({name: 'Model Summary'}, model);

  model.compile({loss: tf.losses.sigmoidCrossEntropy,
                optimizer: tf.train.adam()});
  
  // Train the model using the data, and test on the same data for now
  model.fit(xs, ys, {epochs: 10}).then(() => {
    model.predict(xs).array().then(d => {
      for (let i = 0; i < d.length; i++) {
        console.log(d[i] + ", " + ysorig[i]);
      }
    })
  });


});

