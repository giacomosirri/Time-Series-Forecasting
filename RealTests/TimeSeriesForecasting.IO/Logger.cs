namespace TimeSeriesForecasting.IO
{
    public abstract class Logger<T>
    {
        protected readonly StreamWriter _stream;

        public Logger(string filePath)
        {
            FileStream fs = File.Exists(filePath) ? File.OpenWrite(filePath) : File.Create(filePath);
            _stream = new StreamWriter(fs);
        }

        public void Log(IList<T> list, string message)
        {
            _stream.WriteLine(message);
            list.AsEnumerable().ToList().ForEach(value => _stream.Write(ValueRepresentation(value)));
        }

        protected abstract string ValueRepresentation(T value);

        public void Dispose() => _stream.Close();
    }
}
