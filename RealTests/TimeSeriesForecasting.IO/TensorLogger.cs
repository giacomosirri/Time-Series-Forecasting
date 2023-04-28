using static TorchSharp.torch;

namespace TimeSeriesForecasting.IO
{
    public class TensorLogger
    {
        private StreamWriter _stream;

        public TensorLogger(string filePath) 
        {
            FileStream fs = File.Exists(filePath) ? File.OpenWrite(filePath) : File.Create(filePath);
            _stream = new StreamWriter(fs);
        }

        public void Log(Tensor tensor, string message)
        {
            _stream.WriteLine(message);
            _stream.Write(tensor.ToString(TorchSharp.TensorStringStyle.Default));
        }

        public void Dispose() => _stream.Close();
    }
}
