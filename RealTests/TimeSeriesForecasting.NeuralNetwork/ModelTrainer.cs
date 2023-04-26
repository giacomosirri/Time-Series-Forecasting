using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;

namespace TimeSeriesForecasting.NeuralNetwork
{
    public class ModelTrainer : IModelTrainer
    {
        private const int BatchSize = 32;

        private readonly Module<Tensor, Tensor> _model;
        private readonly Optimizer _optimizer;
        private readonly double _learningRate = 0.1;
        // Type of x (features), type of y (labels) --> type of the result.
        private readonly Loss<Tensor, Tensor, Tensor> _lossFunction;
        private readonly IList<float> _losses = new List<float>();

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

        public ModelTrainer(Module<Tensor, Tensor> model) 
        { 
            _model = model;
            _optimizer = new SGD(_model.parameters(), _learningRate);
            _lossFunction = new MSELoss();
        }

        public void Fit(Tensor x, Tensor y, int epochs)
        {
            Tensor[] batched_x = x.split(BatchSize);
            Tensor[] batched_y = y.split(BatchSize);
            for (int i = 0; i < epochs; i++)
            {
                Tensor? output = null;
                for (int j = 0; j < batched_x.Length; j++)
                {
                    // Compute the loss.
                    output = _lossFunction.forward(_model.forward(batched_x[j]), batched_y[j].flatten(start_dim: 1));
                    // Clear the gradients before doing the back-propagation.
                    _model.zero_grad();
                    // Do back-progatation, which computes all the gradients.
                    output.backward();
                    _optimizer.step();
                }
                _losses.Add(output!.item<float>());
            }
            IsTrained = true;
            LogLosses();
        }

        private void LogLosses()
        {
            _losses.AsEnumerable()
                   .Select((value, index) => (index, value))
                   .ToList()
                   .ForEach(tuple => Console.WriteLine($"Epoch {tuple.index}: MSE = {tuple.value}"));
        }
    }
}
