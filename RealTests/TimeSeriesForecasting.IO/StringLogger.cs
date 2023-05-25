namespace TimeSeriesForecasting.IO
{
    public class StringLogger : Logger<string>
    {
        public StringLogger(string filePath) : base(filePath) {}

        protected override string ValueRepresentation(string value) => value.ToString();
    }
}
