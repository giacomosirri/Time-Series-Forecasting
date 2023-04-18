using System.Data;
using static TorchSharp.torch;

namespace TimeSeriesForecasting.DataProcessing
{
    internal interface IWindowGenerator
    {
        /// <summary>
        /// Splits the data into features and labels.
        /// </summary>
        /// <returns></returns>
        public Tuple<Tensor, Tensor> GenerateWindows<T>(DataTable table);
    }
}
