using System.Data;
using TimeSeriesForecasting;
using Record = TimeSeriesForecasting.IO.Record;

namespace TestTimeSeriesForecasting
{
    public class UnitTestPreprocessor
    {
        private static readonly double Tolerance = 10e-5;
        private readonly DataPreprocessor _preprocessor;

        public UnitTestPreprocessor() 
        {
            IList<Record> records = new List<Record>();
            DateTime? startDate = new DateTime(2010, 10, 10);
            var rnd = new Random(6789);
            for (int i = 0; i < 100; i++)
            {
                IDictionary<string, double?> values = new Dictionary<string, double?>
                {
                    { "A", rnd.NextDouble() * rnd.Next() },
                    { "B", rnd.NextDouble() * rnd.Next() },
                    { "C", rnd.NextDouble() * rnd.Next() },
                    { "D", rnd.NextDouble() * rnd.Next() },
                    { "E", rnd.NextDouble() * rnd.Next() }
                };
                DateTime? date = startDate?.AddDays(i);
                records.Add(new Record(values, date));
            }
            _preprocessor = new DataPreprocessor(records);
        }

        [Fact]
        public void TestNormalization()
        {
            Assert.Throws<ArgumentException>(() => _preprocessor.Normalization = "What?");
            // Normalize the data using Min-Max normalization.
            _preprocessor.Normalization = "Normalization";
            DataTable trainingSet = _preprocessor.GetTrainingSet();
            for (int i = 0; i<trainingSet.Columns.Count; i++) 
            {
                DataColumn col = trainingSet.Columns[i];
                if (col.ColumnName != "Date Time")
                {
                    IDictionary<string, double> stats = GetStats(trainingSet, col.ColumnName, new List<string>() { "Min", "Max" });
                    // The minimum of a column must be 0.
                    Assert.True(Math.Abs(stats["Min"]) < Tolerance);
                    // The maximum of a column must be 1.
                    Assert.True(Math.Abs(stats["Max"] - 1) < Tolerance);
                }
            }
            // Repeat the process for standardization.
            _preprocessor.Normalization = "Standardization";
            DataTable trainingSet2 = _preprocessor.GetTrainingSet();
            for (int i = 0; i < trainingSet2.Columns.Count; i++)
            {
                DataColumn col = trainingSet2.Columns[i];
                if (col.ColumnName != "Date Time")
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