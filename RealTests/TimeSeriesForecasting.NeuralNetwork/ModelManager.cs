using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace TimeSeriesForecasting.NeuralNetwork
{
    public class ModelManager : IModelManager
    {
        /*
         * MaxEpochs and Arrest values can be declared as const, because they are not hyperparameters.
         * Their values are set so as not to affect how the model behaves. In particular, MaxEpochs is 
         * large enough for a reasonably simple model to converge, and Arrest is so close to 0 that 
         * the model can be considered stable when the loss difference between two iterations is smaller 
         * than that value (currently not in use).
         */
        private const int MaxEpochs = 50;
        private const double Arrest = 1e-4;

        private readonly NetworkModel _model;       
        // Type of features, type of labels --> type of the result.
        private readonly Loss<Tensor, Tensor, Tensor> _lossFunction;
        private readonly IList<float> _losses = new List<float>();
        // Hyperparameters
        private readonly double _learningRate = 1e-5;
        private readonly int _batchSize = 64;

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

        public ModelManager(NetworkModel model)
        {
            _model = model;
            _lossFunction = new MSELoss();
        }

        public void Fit(Tensor trainX, Tensor trainY, Tensor validX, Tensor validY)
        {
            TuneHyperparameters(validX, validY);
            Fit(trainX, trainY);
        }

        private void TuneHyperparameters(Tensor x, Tensor y)
        {
            // hyperparameters to be tuned: batch_size and learning_rate
            // (might use a learning rate scheduler instead).
        }

        public void Fit(Tensor trainX, Tensor trainY)
        {
            _model.train();
            var optimizer = new Adam(_model.parameters(), _learningRate);
            Tensor[] batched_x = trainX.split(_batchSize);
            Tensor[] batched_y = trainY.split(_batchSize);
            for (int i = 0; i < MaxEpochs; i++)
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

        public Tensor Predict(Tensor x)
        {
            _model.eval();
            // Disabling autograd gradient calculation speeds up computation.
            using var _ = no_grad();
            Tensor[] batched_x = x.split(_batchSize);
            // currentOutput.shape = [batchSize, outputTimeSteps, outputFeatures].
            Tensor currentOutput = _model.forward(batched_x[0]);
            long start = 0;
            // Control that the stop index is not greater than the number of total batches.
            long stop = Math.Min(x.size(0), _batchSize);
            // Initialize the final output tensor to dirty values, since it is going to be overwritten.
            Tensor y = empty(x.size(0), currentOutput.size(1), currentOutput.size(2));
            // Update the output tensor.
            y.index_copy_(0, arange(start, stop, 1), currentOutput);
            for (int i = 1; i < batched_x.Length; i++)
            {
                currentOutput = _model.forward(batched_x[i]);
                start = _batchSize * i;
                stop = Math.Min(x.size(0), _batchSize * (i + 1));
                // Update the output tensor.
                y.index_copy_(0, arange(start, stop, 1), currentOutput);
            }
            return y;
        }

        public void Save(string directory) => _model.save(Path.Combine(new string[] { directory, "LSTM.model.bin" }));

        public string Summarize() => string.Join(", ", _model.named_modules().Select(module => module.name).ToList());
    }
}
