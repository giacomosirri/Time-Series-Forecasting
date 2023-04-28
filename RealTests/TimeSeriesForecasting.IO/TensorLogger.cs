using static TorchSharp.torch;

namespace TimeSeriesForecasting.IO
{
    public class TensorLogger : Logger<Tensor>
    {
        public TensorLogger(string filePath) : base(filePath) {}

        protected override string ValueRepresentation(Tensor value) => value.ToString(TorchSharp.TensorStringStyle.Default);
    }
}
