namespace TimeSeriesForecasting.IO
{
    public class Record
    {
        private readonly string[] _column_names = new string[]
        {
            "Date Time", "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", "VPmax (mbar)", "VPact (mbar)",
            "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)", "wv (m/s)", "max. wv (m/s)", "wd (deg)"
        };

        public static IDictionary<string, Tuple<double, double>> ValueRanges { get; } = 
            new Dictionary<string, Tuple<double, double>> 
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

        public IDictionary<string, double> Features { get; private set; }
        
        public DateTime TimeStamp { get; set; }
        public double AirPressure { get; set; }
        public double AirTemperature { get; set; }
        public double PotentialTemperature { get; set; }
        public double DewPointTemperature { get; set; }
        public double RelativeHumidity { get; set; }
        public double SaturationWaterVaporPressure { get; set; }
        public double ActualWaterVaporPressure { get; set; }
        public double WaterVaporPressureDeficit { get; set; }
        public double SpecificHumidity { get; set; }
        public double WaterVaporConcentration { get; set; }
        public double AirDensity { get; set; }
        public double WindVelocity { get; set; }
        public double MaximumWindVelocity { get; set; }
        public double WindDirection { get; set; }

        public Record(IDictionary<string, double?> values, DateTime? datetime)
        {
            if ((datetime == null) || values.Values.Where(val => !val.HasValue).Any()) 
            {
                throw new ArgumentNullException("Instantiation of a new record with missing data is not allowed");
            }
            TimeStamp = datetime.Value;
            AirPressure = values[_column_names[1]]!.Value;
            AirTemperature = values[_column_names[2]]!.Value!;
            PotentialTemperature = values[_column_names[3]]!.Value;
            DewPointTemperature = values[_column_names[4]]!.Value;
            RelativeHumidity = values[_column_names[5]]!.Value;
            SaturationWaterVaporPressure = values[_column_names[6]]!.Value;
            ActualWaterVaporPressure = values[_column_names[7]]!.Value;
            WaterVaporPressureDeficit = values[_column_names[8]]!.Value;
            SpecificHumidity = values[_column_names[9]]!.Value;
            WaterVaporConcentration = values[_column_names[10]]!.Value;
            AirDensity = values[_column_names[11]]!.Value;
            WindVelocity = values[_column_names[12]]!.Value;
            MaximumWindVelocity = values[_column_names[13]]!.Value;
            WindDirection = values[_column_names[14]]!.Value;
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
