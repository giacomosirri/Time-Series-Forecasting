using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;

namespace TimeSeriesForecasting.NeuralNetwork
{
    public class ModelTrainer : IModelTrainer
    {
        //private const int BatchSize = 32;

        private readonly Module<Tensor, Tensor> _model;
        private readonly Optimizer _optimizer;
        private readonly double _learningRate;
        private readonly Loss<Tensor, Tensor, Tensor> _loss;

        public ModelTrainer(Module<Tensor, Tensor> model) 
        { 
            _model = model;
            _optimizer = new SGD(_model.parameters(), _learningRate);
            _loss = new MSELoss();
        }

        public void Fit(Tensor x, Tensor y, int epochs)
        {
            for (int i = 0; i < epochs; i++)
            {
                var output = _loss.forward(_model.forward(x), y);
                // Clear the gradients before doing the back-propagation.
                _model.zero_grad();
                // Do back-progatation, which computes all the gradients.
                output.backward();
                _optimizer.step();
            }
        }
    }
}
