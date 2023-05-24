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
        // How much time the last training took.
        public TimeSpan LastTrainingTime { get; private set; }

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
            DateTime start = DateTime.Now;
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
            DateTime end = DateTime.Now;
            LastTrainingTime = end - start;
            IsTrained = true;
        }

        public IDictionary<AccuracyMetric, double> EvaluateAccuracy(Tensor x, Tensor y)
        {
            var dict = new Dictionary<AccuracyMetric, double>();
            Tensor predictedOutput = Predict(x);
            Tensor error = predictedOutput - y;
            dict.Add(AccuracyMetric.MSE, mean(square(error)).item<float>());
            dict.Add(AccuracyMetric.RMSE, Math.Sqrt(mean(square(error)).item<float>()));
            dict.Add(AccuracyMetric.MAE, mean(abs(error)).item<float>());
            dict.Add(AccuracyMetric.MAPE, mean(abs(error / y)).item<float>());
            var r2 = 1 - sum(square(error)).item<float>() / sum(square(y - mean(y))).item<float>();
            var adjustedR2 = 1 - (1 - r2) * (x.size(0) - 1) / (x.size(0) - x.size(2) - 1);
            dict.Add(AccuracyMetric.R2, adjustedR2);
            return dict;
        }

        public Tensor Predict(Tensor x)
        {
            _model.eval();
            // Disabling autograd gradient calculation speeds up computation.
            using var _ = no_grad();
            Tensor[] batched_x = x.split(_batchSize);
            // output.shape = [batchSize, outputTimeSteps, outputFeatures].
            Tensor output = _model.forward(batched_x[0]);
            long start = 0;
            // Control that the stop index is not greater than the number of total batches.
            long stop = Math.Min(x.size(0), _batchSize);
            // Initialize the output tensor to zeros.
            Tensor y = zeros(x.size(0), output.size(1), output.size(2));
            // Update the output tensor.
            y.index_copy_(0, arange(start, stop, 1), output);
            Console.WriteLine(y[0][0][0].item<float>());
            for (int i = 1; i < batched_x.Length; i++)
            {
                output = _model.forward(batched_x[i]);
                start = _batchSize * i;
                stop = Math.Min(x.size(0), _batchSize * (i + 1));
                // Update the output tensor.
                y.index_copy_(0, arange(start, stop, 1), output);
            }
            return output;
        }

        public void Save(string directory) => _model.save(Path.Combine(new string[] { directory, "LSTM.model.bin" }));
    }
}
