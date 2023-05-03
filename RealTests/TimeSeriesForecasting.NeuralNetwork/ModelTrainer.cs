using TimeSeriesForecasting.IO;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace TimeSeriesForecasting.NeuralNetwork
{
    public class ModelTrainer : IModelTrainer
    {
        /*
         * MaxEpochs and Arrest values can be declared as const, because they are not hyperparameters.
         * Their values are set so as not to affect how the model behaves. In particular, MaxEpochs is 
         * large enough for a reasonably simple model to converge, and Arrest is so close to 0 that 
         * the model can be considered stable when the loss difference between two iterations is smaller 
         * than that value (currently not in use).
         */
        private const int MaxEpochs = 100;
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
            _learningRate = 1e-6;
            _batchSize = 32;
        }

        public void Fit(Tensor x, Tensor y)
        {
            _model.train();
            var optimizer = new Adam(_model.parameters(), _learningRate);
            Tensor[] batched_x = x.split(_batchSize);
            Tensor[] batched_y = y.split(_batchSize);
            Tensor previousOutput = tensor(float.MaxValue);
            for (int i = 0; i < MaxEpochs; i++)
            {
                Tensor output = empty(1);
                float epochLoss = 0.0f;
                for (int j = 0; j < batched_x.Length; j++)
                {
                    // Compute the loss.
                    output = _lossFunction.forward(_model.forward(batched_x[j]), batched_y[j].flatten(start_dim: 1));                    
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
            // Log the computed losses to file.
            _logger.Log(_losses.AsEnumerable().Select((value, index) => (index, value)).ToList(), 
                $"MSE with learning rate {_learningRate} and batch size {_batchSize}:");
            _logger.Dispose();
        }

        public IDictionary<AccuracyMetric, double> EvaluateAccuracy(Tensor x, Tensor y)
        {
            _model.eval();
            var dict = new Dictionary<AccuracyMetric, double>();
            Tensor[] batched_x = x.split(_batchSize);
            int batches = batched_x.Length;
            Tensor expectedOutput = y.squeeze();
            Tensor predictedOutput = empty(x.size(0));
            for (int i = 0; i < batches; i++)
            {
                long start = _batchSize*i;
                // Control that the stop index is not greater than the number of total batches.
                long stop = Math.Min(x.size(0), _batchSize*(i+1));
                // Create the predicted output tensor.
                predictedOutput.index_copy_(0, arange(start, stop, 1), _model.forward(batched_x[i]).squeeze());
            }
            Tensor error = predictedOutput - expectedOutput;
            dict.Add(AccuracyMetric.MSE, mean(square(error)).item<float>());
            dict.Add(AccuracyMetric.RMSE, Math.Sqrt(mean(square(error)).item<float>()));
            dict.Add(AccuracyMetric.MAE, mean(abs(error)).item<float>());
            dict.Add(AccuracyMetric.MAPE, mean(abs(error / expectedOutput)).item<float>());
            dict.Add(AccuracyMetric.R2, 0);
            return dict;
        }

        public Tensor Predict(Tensor x)
        {
            throw new NotImplementedException();
        }
    }
}
