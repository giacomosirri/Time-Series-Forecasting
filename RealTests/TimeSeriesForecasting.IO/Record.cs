namespace TimeSeriesForecasting.IO
{
    public class Record
    {
        public static IDictionary<string, Tuple<double, double>> ValueRanges { get; } = new Dictionary<string, Tuple<double, double>> 
        {
            { "mbar", Tuple.Create(0.0, 1150.0) },
            { "degC", Tuple.Create(-50.0, 50.0) },
            { "K", Tuple.Create(220.0, 325.0) },
            { "%", Tuple.Create(0.0, 100.0) },
            { "g/kg", Tuple.Create(0.0, 30.0) },
            { "mmol/mol", Tuple.Create(0.0, 40.0) },
            { "g/m**3", Tuple.Create(800.0, 1500.0) },
            { "m/s", Tuple.Create(0.0, 70.0) },
            { "deg", Tuple.Create(0.0, 360.0) }
        };

        /// <summary>
        /// Returns the name of the column that acts as index for a record, similar to an index in a Pandas' Series or DataFrame.
        /// Such column must be of type <see cref="DateTime"/>, as these record represent observations in a time series.
        /// </summary>
        public static string Index { get; } = "Date Time";

        public IDictionary<string, double> Features { get; private set; }
        
        public DateTime TimeStamp { get; }

        public Record(IDictionary<string, double?> values, DateTime? datetime)
        {
            if ((datetime == null) || values.Values.Where(val => !val.HasValue).Any()) 
            {
                throw new ArgumentNullException("Instantiation of a new record with missing data is not allowed");
            }
            TimeStamp = datetime.Value;
            Features = values.Keys.Zip(values.Values.Select(elem => elem!.Value).ToList(), 
                (k,v) => new {Key = k, Value = v}).ToDictionary(x => x.Key, x => x.Value);
        }
        
        public double[] GetNumericValues() => Features.Values.ToArray();

        public static string? GetUnitOfMeasureFromFeatureName(string feature)
        {
            try
            {
                var str = feature.Split('(').Last().Split(")").First();
                return str == "" || str == feature ? null : str;
            }
            catch (InvalidOperationException)
            {
                return null;
            }
        }

        public override string ToString()
        {
            return $"{TimeStamp}:\n" + string.Join(",", Features.Select(i => $"{i.Key}: {i.Value}")) + "\n";
        }
    }
}
