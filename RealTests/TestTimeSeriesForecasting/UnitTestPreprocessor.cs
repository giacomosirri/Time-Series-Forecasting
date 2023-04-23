using System.Data;
using TimeSeriesForecasting;
using Record = TimeSeriesForecasting.IO.Record;
using static TimeSeriesForecasting.DataPreprocessor;

namespace TestTimeSeriesForecasting
{
    public class UnitTestPreprocessor
    {
        private const double Tolerance = 10e-5;
        private const int LowerBound = -10;
        private const int UpperBound = 51;

        private readonly DataPreprocessor _preprocessor;

        public UnitTestPreprocessor() 
        {
            IList<Record> records = new List<Record>();
            // 10/10/2010 20:00:00
            DateTime? currentDate = new DateTime(2010, 10, 10, 20, 0, 0);
            var rnd = new Random(6789);
            for (int i = 0; i < 10000; i++)
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
            _preprocessor = new DataPreprocessor(records);
        }

        [Fact]
        public void TestMinMaxNormalization()
        {
            // Normalize the data using Min-Max normalization.
            _preprocessor.Normalization = NormalizationMethod.MIN_MAX_NORMALIZATION;
            DataTable trainingSet = _preprocessor.GetTrainingSet();
            for (int i = 0; i<trainingSet.Columns.Count; i++) 
            {
                DataColumn col = trainingSet.Columns[i];
                if (col.ColumnName != Record.Index)
                {
                    IDictionary<string, double> stats = GetStats(trainingSet, col.ColumnName, new List<string>() { "Min", "Max" });
                    // The minimum of a column must be 0.
                    Assert.True(Math.Abs(stats["Min"]) < Tolerance);
                    // The maximum of a column must be 1.
                    Assert.True(Math.Abs(stats["Max"] - 1) < Tolerance);
                }
            }
        }

        [Fact]
        public void TestStandardization() 
        {
            // Normalize the data using standardization.
            _preprocessor.Normalization = NormalizationMethod.STANDARDIZATION;
            DataTable trainingSet2 = _preprocessor.GetTrainingSet();
            for (int i = 0; i < trainingSet2.Columns.Count; i++)
            {
                DataColumn col = trainingSet2.Columns[i];
                if (col.ColumnName != Record.Index)
                {
                    IDictionary<string, double> stats = GetStats(trainingSet2, col.ColumnName, new List<string>() { "Avg", "StDev" });
                    // The mean of any column after standardization must be 0.
                    Assert.True(Math.Abs(stats["Avg"]) < Tolerance);
                    // The standard deviation of any column after standardization must be 0.
                    Assert.True(Math.Abs(stats["StDev"] - 1) < Tolerance);
                }
            }
        }

        private static IDictionary<string, double> GetStats(DataTable dt, string col, IList<string> operations)
        {
            IDictionary<string, double> stats = new Dictionary<string, double>();
            operations.AsEnumerable().ToList().ForEach(op => stats.Add(op, Convert.ToDouble(dt.Compute($"{op}([{col}])", ""))));
            return stats;
        }
    }
}