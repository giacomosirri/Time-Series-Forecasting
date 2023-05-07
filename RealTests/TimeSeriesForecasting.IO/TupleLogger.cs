namespace TimeSeriesForecasting.IO
{
    public class TupleLogger<T> : Logger<(T key, float value)>
    {
        public TupleLogger(string filePath) : base(filePath) { }

        protected override string ValueRepresentation((T key, float value) value) => 
            $"After {value.key,-2} epochs\t: {value.value:F4}";
    }
}