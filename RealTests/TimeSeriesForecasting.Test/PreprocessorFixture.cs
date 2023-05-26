using TimeSeriesForecasting;
using Record = TimeSeriesForecasting.IO.Record;

namespace TestTimeSeriesForecasting
{
    public class PreprocessorFixture : IDisposable
    {
        // Do not change this field: 14400 = 100 days * 24 hours * 6 timesteps/hour
        private const int TimeSteps = 14400;
        private const int LowerBound = -10;
        private const int UpperBound = 51;

        public double Tolerance { get; } = 10e-5;
        public DateTime? FirstDatasetDate { get; } = new DateTime(2010, 10, 10, 20, 0, 0);
        public DateTime? LastDatasetDate { get; } = new DateTime(2011, 01, 18, 19, 0, 0);
        public DataPreprocessor Preprocessor { get; set; }

        public PreprocessorFixture()
        {
            IList<Record> records = new List<Record>();
            DateTime? currentDate = FirstDatasetDate;
            var rnd = new Random(6789);
            for (int i = 0; i < TimeSteps; i++)
            {
                IDictionary<string, double?> values = new Dictionary<string, double?>
                {
                    { "A", rnd.NextDouble() * rnd.Next(LowerBound, UpperBound) },
                    { "B", rnd.NextDouble() * rnd.Next(LowerBound, UpperBound) },
                    { "C", rnd.NextDouble() * rnd.Next(LowerBound, UpperBound) },
                    { "D", rnd.NextDouble() * rnd.Next(LowerBound, UpperBound) },
                    { "E", rnd.NextDouble() * rnd.Next(LowerBound, UpperBound) }
                };
                records.Add(new Record(values, currentDate));
                currentDate = currentDate.Value.AddMinutes(10);
            }
            Preprocessor = new DataPreprocessorBuilder().Build(records);
        }

        public void Dispose()
        {
            Preprocessor.Dispose();
        }
    }

    [CollectionDefinition("Preprocessor collection")]
    public class PreprocessorCollection : ICollectionFixture<PreprocessorFixture> {}
}

