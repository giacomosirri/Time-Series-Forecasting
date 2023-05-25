using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TimeSeriesForecasting.ANN
{
    /// <summary>
    /// This class is the base class for all the neural networks implemented in this project.
    /// It is a subclass of <see cref="Module{Tensor, Tensor}"/> as all networks work with 
    /// <see cref="Tensor"/>s as inputs and outputs.
    /// </summary>
    public abstract class NeuralNetwork : Module<Tensor, Tensor>
    {
        protected NeuralNetwork(string name) : base(name)
        {
            /*
             * When the seed is set manually, the weights of the models are always initialized to the same values.
             * This allows comparisons between different executions with different hyperparameters.
             * Using a specific parameters initialization function instead of setting them randomly would be a better choice.
             */
            manual_seed(42);
        }
    }
}
