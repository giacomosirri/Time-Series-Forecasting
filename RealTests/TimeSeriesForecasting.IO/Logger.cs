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

        public void Log(T value, string message) 
        {
            using StreamWriter writer = _stream;
            writer.WriteLine(message);
            writer.WriteLine(ValueRepresentation(value));
        }

        public void Log(IList<T> list, string message)
        {
            using StreamWriter writer = _stream;
            writer.WriteLine(message);
            list.AsEnumerable().ToList().ForEach(value => writer.WriteLine(ValueRepresentation(value)));
        }

        protected abstract string ValueRepresentation(T value);
    }
}
