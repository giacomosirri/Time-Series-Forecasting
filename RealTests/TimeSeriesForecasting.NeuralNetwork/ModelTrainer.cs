using TimeSeriesForecasting.IO;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;

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
        private const double Arrest = 10e-5;

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
            _batchSize = 64;
        }

        public void Fit(Tensor x, Tensor y)
        {
            int i = 0;
            var optimizer = new Adam(_model.parameters(), _learningRate);
            Tensor[] batched_x = x.split(_batchSize);
            Tensor[] batched_y = y.split(_batchSize);
            Tensor previousOutput = tensor(float.MaxValue);
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
                _logger.LogComment($"The training converges after {i} epochs.");
            }
            _logger.Dispose();
        }

        public IList<double> TestModelPerformance(Tensor x, Tensor y, IList<string> metrics)
        {
            throw new NotImplementedException();
        }

        public Tensor Predict(Tensor x)
        {
            throw new NotImplementedException();
        }
    }
}
