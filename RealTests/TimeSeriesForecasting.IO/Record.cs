using Newtonsoft.Json.Linq;

namespace TimeSeriesForecasting.IO
{
    public class Record
    {
        /*
         * This dictionary maps a feature name to a pair of doubles that represent the minimum and maximum
         * values that can be assumed by an instance of that feature.
         */
        public static IDictionary<string, (double? minValue, double? maxValue)> ValueRanges { get; } =
            new Dictionary<string, (double? minValue, double? maxValue)>();

        /* 
         * Returns the name of the column that acts as index for a record, similar to an index in a Pandas' Series or DataFrame.
         * Such column must be of type DateTime, as these record represent time steps in a time series.
         */
        public static string Index { get; set; } = "";

        public IDictionary<string, double> Features { get; private set; }
        
        public DateTime TimeStamp { get; }

        public Record(IDictionary<string, double?> values, DateTime? datetime)
        {
            if ((datetime == null) || values.Values.Where(val => !val.HasValue).Any()) 
            {
                throw new ArgumentNullException("Instantiation of a new record with missing data is not allowed.");
            }
            TimeStamp = datetime.Value;
            Features = values.Keys.Zip(values.Values.Select(elem => elem!.Value).ToList(), 
                (k,v) => new {Key = k, Value = v}).ToDictionary(x => x.Key, x => x.Value);
        }

        /*
         * Reads the features value ranges from a json file. Returns true if the operation is successful, false otherwise.
         */
        public static bool ReadValueRangesFromJsonFile(string filePath)
        {
            try
            {
                JObject obj = JObject.Parse(File.ReadAllText(filePath));
                foreach (var key in obj)
                {
                    string feature = key.Key;
                    if (key.Value != null)
                    {
                        double? minValue = (double?)key.Value[0];
                        double? maxValue = (double?)key.Value[1];
                        ValueRanges[feature] = (minValue, maxValue);
                    }
                    else return false;
                }
                return true;
            } 
            catch (Exception)
            {
                return false;
            }
        }

        public override string ToString()
        {
            return $"{TimeStamp}:\n" + string.Join("\n", Features.Select(i => $"{i.Key}: {i.Value}")) + "\n";
        }
    }
}
