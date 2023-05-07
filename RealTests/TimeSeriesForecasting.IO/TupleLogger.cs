namespace TimeSeriesForecasting.IO
{
    public class TupleLogger<K, V> : Logger<(K key, V value)>
    {
        public TupleLogger(string filePath) : base(filePath) { }

        protected override string ValueRepresentation((K key, V value) value) => $"{value.key,-3} \t: {value.value:F4}";
    }
}