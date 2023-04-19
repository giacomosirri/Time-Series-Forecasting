﻿using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TimeSeriesForecasting.NeuralNetwork
{
    public class Baseline : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> _linear;

        public Baseline(long inputObservations, long inputFeatures, long outputObservations, long outputFeatures) 
            : base(nameof(Baseline))
        {
            RegisterComponents();
            _linear = Linear(inputObservations * inputFeatures, outputObservations * outputFeatures);
        }

        public override Tensor forward(Tensor input)
        {
            // Input tensor is a 3D Tensor of shape (batch_size, observations, features).
            Tensor flattenedInput = input.flatten(start_dim: 1);
            // Flattened input Tensor has shape (batch_size, observations * features), so it is compatible with the linear layer.
            return _linear.forward(flattenedInput).type_as(input);
        }
    }
}