using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TimeSeriesForecasting.NeuralNetwork
{
    /// <summary>
    /// This class is the base class for all the neural networks implemented in this project.
    /// It is a subclass of <see cref="Module{Tensor, Tensor}"/> as all networks work with 
    /// <see cref="Tensor"/>s as inputs and outputs. 
    /// Other than the <see cref="Module{T, TResult}.forward(Tensor)"/> method, the other 
    /// method that subclasses need to implement is <see cref="NetworkModel.LogState(string)"/>,
    /// which outputs the current model parameters values to file.
    /// </summary>
    public abstract class NetworkModel : Module<Tensor, Tensor>
    {
        protected NetworkModel(string name) : base(name) {}

        public abstract void LogState(string message);
    }
}
