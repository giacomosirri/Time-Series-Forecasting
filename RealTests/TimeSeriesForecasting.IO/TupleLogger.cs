namespace TimeSeriesForecasting.IO
{
    public class TupleLogger<T> : Logger<(T key, float value)>
    {
        public TupleLogger(string filePath) : base(filePath) { }

        protected override string ValueRepresentation((T key, float value) value) => $"{value.key,-2} \t: {value.value:F4}";
    }
}