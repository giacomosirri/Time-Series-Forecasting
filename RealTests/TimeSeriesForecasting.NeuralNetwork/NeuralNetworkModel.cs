using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace TimeSeriesForecasting.ANN
{
    public class NeuralNetworkModel : INeuralNetworkModel
    {
        // All the loss functions that can be used in the context of this class.
        public enum LossFunction
        {
            MSE, L1
        }

        // All the optimizers that can be used in the context of this class.
        public enum Optimizer
        {
            SGD, ADAM, ADAMAX, ADAGRAD, RMSPROP
        }

        private readonly NeuralNetwork _model;
        private readonly Loss<Tensor, Tensor, Tensor> _lossFunction;
        private readonly Optimizer _optimizer;
        private readonly IList<float> _losses = new List<float>();
        private int _lastUsedBatchSize;

        public bool IsTrained { get; private set; } = false;

        public IList<float> LossProgress
        {
            get
            {
                if (IsTrained)
                {
                    return _losses;
                }
                else
                {
                    throw new InvalidOperationException("Loss progress is unavailable because the model has never been trained.");
                }
            }
        }

        private NeuralNetworkModel(NeuralNetwork model, LossFunction loss, Optimizer optimizer)
        {
            _model = model;
            _lossFunction = (loss == LossFunction.MSE) ? new MSELoss() : new L1Loss();
            _optimizer = optimizer;
        }


        public void Fit(Tensor trainX, Tensor trainY, Tensor valX, Tensor valY, int epochs, int batchSize, double learningRate)
        {
            Fit(trainX, trainY, epochs, batchSize, learningRate);
        }

        public void Fit(Tensor trainX, Tensor trainY, int epochs, int batchSize, double learningRate)
        {
            _model.train();
            optim.Optimizer optimizer = GetOptimizer(learningRate);
            Tensor[] batched_x = trainX.split(batchSize);
            Tensor[] batched_y = trainY.split(batchSize);
            for (int i = 0; i < epochs; i++)
            {
                float epochLoss = 0.0f;
                for (int j = 0; j < batched_x.Length; j++)
                {
                    // Compute the loss.
                    Tensor predicted = _model.forward(batched_x[j]);
                    Tensor expected = batched_y[j];
                    Tensor output = _lossFunction.forward(predicted, expected);                    
                    // Sum the loss for this batch to the total loss for this epoch.
                    epochLoss += output.item<float>();
                    // Clear the gradients before doing the back-propagation.
                    _model.zero_grad();
                    // Do back-progatation, which computes all the gradients.
                    output.backward();
                    // Modifies the weights and biases to reduce the loss.
                    optimizer.step();
                }
                // Add the average loss for all the batches in this epoch to the list of losses.
                _losses.Add(epochLoss / batched_x.Length);
            }
            IsTrained = true;
            _lastUsedBatchSize = batchSize;
        }

        private optim.Optimizer GetOptimizer(double learningRate)
        {
            return _optimizer switch
            {
                Optimizer.SGD => new SGD(_model.parameters(), learningRate),
                Optimizer.ADAM => new Adam(_model.parameters(), learningRate),
                Optimizer.ADAMAX => new Adamax(_model.parameters(), learningRate),
                Optimizer.ADAGRAD => new Adagrad(_model.parameters(), learningRate),
                Optimizer.RMSPROP => new RMSProp(_model.parameters(), learningRate),
                _ => new SGD(_model.parameters(), learningRate),
            };
        }

        public IDictionary<AccuracyMetric, IList<double>> Evaluate(Tensor predictedOutput, Tensor expectedOutput)
        {
            var dict = new Dictionary<AccuracyMetric, IList<double>>()
            {
                { AccuracyMetric.MSE, new List<double>() },
                { AccuracyMetric.RMSE, new List<double>() },
                { AccuracyMetric.MAE, new List<double>() },
                { AccuracyMetric.MAPE, new List<double>() },
                { AccuracyMetric.R2, new List<double>() }
            };
            long samples = predictedOutput.size(0);
            long outputTimeSteps = predictedOutput.size(1);
            long outputFeatures = predictedOutput.size(2);
            Tensor expected = empty(samples * outputTimeSteps, outputFeatures);
            Tensor predicted = empty(samples * outputTimeSteps, outputFeatures);
            // Iterate over all the samples to properly fill expected and predicted tensors.
            for (int i = 0; i < samples; i++)
            {
                long start = outputTimeSteps * i;
                long stop = outputTimeSteps * (i + 1);
                Tensor slice = arange(start, stop, 1); 
                expected.index_copy_(0, slice, expectedOutput[i]);
                predicted.index_copy_(0, slice, predictedOutput[i]);
            }
            // Iterate over all the features to calculate the different metrics using the precalculated tensors.
            for (int i = 0; i < outputFeatures; i++)
            {
                Tensor currentFeatureExpectedValues = expected.swapaxes(0, 1)[i];
                Tensor currentFeaturePredictedValues = predicted.swapaxes(0, 1)[i];
                Tensor error = currentFeaturePredictedValues - currentFeatureExpectedValues;
                dict[AccuracyMetric.MSE].Add(mean(square(error)).item<float>());
                dict[AccuracyMetric.RMSE].Add(Math.Sqrt(mean(square(error)).item<float>()));
                dict[AccuracyMetric.MAE].Add(mean(abs(error)).item<float>());
                dict[AccuracyMetric.MAPE].Add(mean(abs(error / expectedOutput)).item<float>());
                var r2 = 1 - sum(square(error)).item<float>() / sum(square(expectedOutput - mean(expectedOutput))).item<float>();
                dict[AccuracyMetric.R2].Add(r2);
            }
            return dict;
        }

        public IDictionary<AccuracyMetric, IList<double>> PredictAndEvaluate(Tensor x, Tensor expectedOutput)
        {
            Tensor predictedOutput = Predict(x);
            return Evaluate(predictedOutput, expectedOutput);
        }

        public Tensor Predict(Tensor x) => Predict(x, _lastUsedBatchSize);

        public Tensor Predict(Tensor x, int batchSize)
        {
            if (!IsTrained)
            {
                throw new InvalidOperationException("The model cannot be used to predict new values if it has not been trained yet.");
            }
            _model.eval();
            // Disabling autograd gradient calculation speeds up computation.
            using var _ = no_grad();
            Tensor[] batched_x = x.split(batchSize);
            // currentOutput.shape = [batchSize, outputTimeSteps, outputFeatures].
            Tensor currentOutput = _model.forward(batched_x[0]);
            long start = 0;
            // Control that the stop index is not greater than the number of total batches.
            long stop = Math.Min(x.size(0), batchSize);
            // Initialize the final output tensor to dirty values, since it is going to be overwritten.
            Tensor y = empty(x.size(0), currentOutput.size(1), currentOutput.size(2));
            // Update the output tensor.
            y.index_copy_(0, arange(start, stop, 1), currentOutput);
            for (int i = 1; i < batched_x.Length; i++)
            {
                currentOutput = _model.forward(batched_x[i]);
                start = batchSize * i;
                stop = Math.Min(x.size(0), batchSize * (i + 1));
                // Update the output tensor.
                y.index_copy_(0, arange(start, stop, 1), currentOutput);
            }
            return y;
        }

        public void Save(string directory) => _model.save(Path.Combine(new string[] { directory, "LSTM.model.bin" }));

        public string Summarize() => string.Join(", ", _model.named_modules().Select(module => module.name).ToList());
    }
}
