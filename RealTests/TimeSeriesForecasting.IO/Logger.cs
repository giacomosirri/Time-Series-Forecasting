namespace TimeSeriesForecasting.IO
{
    public abstract class Logger<T>
    {
        protected readonly StreamWriter _stream;

        public Logger(string filePath)
        {
            FileStream fs = File.Exists(filePath) ? File.Open(filePath, FileMode.Append) : File.Create(filePath);
            _stream = new StreamWriter(fs);
        }

        public void Log(T value, string message)
        {
            _stream.WriteLine(message);
            _stream.Write(ValueRepresentation(value));
        }

        protected abstract string ValueRepresentation(T value);

        public void Dispose() => _stream.Close();
    }
}
