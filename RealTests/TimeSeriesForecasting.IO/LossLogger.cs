namespace TimeSeriesForecasting.IO
{
    public class LossLogger : Logger<(int index, float value)>
    {
        public LossLogger(string filePath) : base(filePath) { }

        protected override string ValueRepresentation((int index, float value) value) => $"Epoch {value.index}: Loss = {value.value}";
    }
}