﻿using TimeSeriesForecasting.IO;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TimeSeriesForecasting.NeuralNetwork
{
    public class ModelTrainer : IModelTrainer
    {
        /*
         * MaxEpochs and Arrest values can be declared as const, because they are not hyperparameters.
         * Their values are set so as not to affect how the model behaves. In particular, MaxEpochs is 
         * large enough for a reasonably simple model to converge, and Arrest is so close to 0 that 
         * the model can be considered stable when the loss difference between two iterations is smaller 
         * than that value.
         */
        private const int MaxEpochs = 250;
        private const double Arrest = 1e-4;

        private readonly NetworkModel _model;       
        // Type of x (features), type of y (labels) --> type of the result.
        private readonly Loss<Tensor, Tensor, Tensor> _lossFunction;
        private readonly IList<float> _losses = new List<float>();
        private readonly LossLogger _logger;
        // Hyperparameters
        private double _learningRate;
        private int _batchSize;

        public bool IsTrained { get; private set; } = false;
        public float CurrentLoss
        { 
            get
            {
                if (IsTrained)
                {
                    // Return the final loss of the trained model.
                    return _losses[^1];
                }
                else
                {
                    throw new InvalidOperationException("Current loss is unavailable because the model has never been trained.");
                }
            } 
        }

        public ModelTrainer(NetworkModel model, string filePath)
        {
            _model = model;
            _lossFunction = new MSELoss();
            _logger = new LossLogger(filePath);
        }

        public void TuneHyperparameters(Tensor x, Tensor y)
        {
            _learningRate = 1e-5;
            _batchSize = 64;
        }

        public void Fit(Tensor x, Tensor y)
        {
            _model.train();
            var optimizer = new Adam(_model.parameters(), _learningRate);
            Tensor[] batched_x = x.split(_batchSize);
            Tensor[] batched_y = y.split(_batchSize);
            Tensor previousOutput = tensor(float.MaxValue);
            int i = 0;
            for (; i < MaxEpochs; i++)
            {
                Tensor output = empty(1);
                for (int j = 0; j < batched_x.Length; j++)
                {
                    // Compute the loss.
                    output = _lossFunction.forward(_model.forward(batched_x[j]), batched_y[j].flatten(start_dim: 1));
                    // Clear the gradients before doing the back-propagation.
                    _model.zero_grad();
                    // Do back-progatation, which computes all the gradients.
                    output.backward();
                    optimizer.step();
                }
                _losses.Add(output.item<float>());
                if (Math.Abs(previousOutput.item<float>() - output.item<float>()) < Arrest)
                {
                    break;
                }
                else
                {
                    previousOutput = output;
                }
            }
            IsTrained = true;
            // Log the computed losses to file.
            _logger.Log(_losses.AsEnumerable().Select((value, index) => (index, value)).ToList(), 
                $"MSE with learning rate {_learningRate} and batch size {_batchSize}:");
            if (i < MaxEpochs)
            {
                _logger.LogComment($"The training converges after {i+1} epochs.");
            }
            _logger.Dispose();
        }

        public IDictionary<AccuracyMetric, double> EvaluateAccuracy(Tensor x, Tensor y)
        {
            _model.eval();
            var dict = new Dictionary<AccuracyMetric, double>();
            double mae = 0, mse = 0, r2 = 0;
            Tensor[] batched_x = x.split(_batchSize);
            Tensor[] batched_y = y.split(_batchSize);
            long inputs = x.shape[0];
            for (int i = 0; i < batched_x.Length; i++)
            {
                Tensor predictedOutput = _model.forward(batched_x[i]);
                Tensor expectedOutput = batched_y[i].flatten(start_dim: 1);
                // Compute the Mean Absolute Error over this batch.
                mae += functional.l1_loss(predictedOutput, expectedOutput).item<float>();
                // Compute the Mean Squared Error over this batch.
                mse += functional.mse_loss(predictedOutput, expectedOutput).item<float>();
                // Compute the mean of the expected outputs.
                var expectedOutputMean = mean(expectedOutput);
                // Compute the total sum of squares (TSS).
                var tss = sum(pow(expectedOutput - expectedOutputMean, 2)).item<float>();
                // Compute the residual sum of squares (RSS).
                var rss = sum(pow(expectedOutput - predictedOutput, 2)).item<float>();
                // Compute the R-squared over this batch.
                r2 += 1 - rss / tss;
            }
            dict.Add(AccuracyMetric.MSE, mse / inputs);
            dict.Add(AccuracyMetric.RMSE, Math.Sqrt(mse / inputs));
            dict.Add(AccuracyMetric.MAE, mae / inputs);
            dict.Add(AccuracyMetric.MAPE, mae / inputs / y.flatten(start_dim: 1).abs().mean().item<float>());
            dict.Add(AccuracyMetric.R2, r2 / inputs);
            return dict;
        }

        public Tensor Predict(Tensor x)
        {
            throw new NotImplementedException();
        }
    }
}
