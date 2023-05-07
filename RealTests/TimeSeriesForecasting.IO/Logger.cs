namespace TimeSeriesForecasting.IO
{
    public abstract class Logger<T>
    {
        protected readonly StreamWriter _stream;

        public Logger(string filePath)
        {
            FileStream fs = File.Exists(filePath) ? File.OpenWrite(filePath) : File.Create(filePath);
            _stream = new StreamWriter(fs);
            _stream.BaseStream.SetLength(0);
            _stream.Flush();
        }

        public void Prepare(T value, string message) 
        {
            if (message != null)
            {
                _stream.WriteLine(message);
            }
            _stream.WriteLine(ValueRepresentation(value));
        }

        public void Prepare(IList<T> list, string message)
        {
            if (message != null)
            {
                _stream.WriteLine(message);
            }
            list.AsEnumerable().ToList().ForEach(value => _stream.WriteLine(ValueRepresentation(value)));
        }

        protected abstract string ValueRepresentation(T value);

        public void Write() => _stream.Close();
    }
}
