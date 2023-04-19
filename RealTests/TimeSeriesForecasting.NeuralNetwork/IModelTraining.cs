﻿using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;

namespace TimeSeriesForecasting.NeuralNetwork
{
    internal interface IModelTraining
    {
        Loss<Tensor, Tensor, Tensor> Loss { get; set; }
        Optimizer Optimizer { get; set; }

        public void Fit(Tensor x, Tensor y, int epochs);
    }
}
