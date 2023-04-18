using System.Data;
using static TorchSharp.torch;

namespace RNN_Model_Creation
{
    internal interface IWindowGenerator
    {
        /// <summary>
        /// Splits the data into features and labels.
        /// </summary>
        /// <returns></returns>
        public Tuple<Tensor, Tensor> GenerateWindows<T>(DataTable table);

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CustomDataset MakeDataset();
    }
}
