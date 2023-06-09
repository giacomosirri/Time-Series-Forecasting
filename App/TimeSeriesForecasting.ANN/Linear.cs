﻿using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TimeSeriesForecasting.ANN
{
    public class Linear : NeuralNetwork
    {
        private readonly TorchSharp.Modules.Linear _linear;

        /*
         * Create a new RNN model with weights and biases initialized with this distribution:
         * U(−sqrt(k), sqrt(k)), where k = 1 / input_features.
         */
        public Linear(long inputTimeSteps, long inputFeatures, long outputTimeSteps, long outputFeatures) 
            : base(nameof(Linear))
        {
            _linear = Linear(inputTimeSteps * inputFeatures, outputTimeSteps * outputFeatures);
            RegisterComponents();
        }

        /*
         * Initialize the model loading its weights and biases from the given file.
         */
        public Linear(long inputTimeSteps, long inputFeatures, long outputTimeSteps, long outputFeatures, string path)
            : this(inputTimeSteps, inputFeatures, outputTimeSteps, outputFeatures) => load(path);

        public override Tensor forward(Tensor input)
        {
            /* 
             * Input tensor is a 3D Tensor of shape (batch_size, time_steps, features).
             * This tensor is flattened before being fed to the network, mainly because 
             * the network will then train faster and be more stable.
             */
            Tensor flattenedInput = input.flatten(start_dim: 1);
            // Flattened input Tensor has shape (batch_size, time_steps * features).
            return _linear.forward(flattenedInput);
        }
    }
}