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
            // 14400 = 100 days * 24 hours * 6 observations/hour
            for (int i = 0; i < 14400; i++)
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
            DataTable table = _preprocessor.GetTrainingSet();
            for (int i = 0; i < table.Columns.Count; i++) 
            {
                string colName = table.Columns[i].ColumnName;
                if (colName != Record.Index)
                {
                    // Calculate the minimum value of the "Value" column.
                    double minValue = table.AsEnumerable().Min(row => row.Field<double>(colName));
                    // Calculate the maximum value of the "Value" column.
                    double maxValue = table.AsEnumerable().Max(row => row.Field<double>(colName));
                    // The minimum of any column must be 0.
                    Assert.True(Math.Abs(minValue) < Tolerance);
                    // The maximum of any column must be 1.
                    Assert.True(Math.Abs(maxValue - 1) < Tolerance);
                }
            }
        }

        [Fact]
        public void TestStandardization() 
        {
            // Normalize the data using standardization.
            _preprocessor.Normalization = NormalizationMethod.STANDARDIZATION;
            DataTable table = _preprocessor.GetTrainingSet();
            for (int i = 0; i < table.Columns.Count; i++)
            {
                string colName = table.Columns[i].ColumnName;
                if (colName != Record.Index)
                {
                    // Calculate the average of the "colName" column.
                    double average = table.AsEnumerable().Average(row => row.Field<double>(colName));
                    // Calculate the standard deviation of the "colName" column.
                    double stdev = Math.Sqrt(table.AsEnumerable().Average(row => Math.Pow(row.Field<double>(colName) - average, 2)));
                    // The mean of any column after standardization must be 0.
                    Assert.True(average < Tolerance);
                    // The standard deviation of any column after standardization must be 1.
                    Assert.True(Math.Abs(stdev - 1) < Tolerance);
                }
            }
        }

        [Fact]
        public void TestDateRange()
        {
            Assert.Equal(new DateTime(2010, 10, 10, 20, 0, 0), _preprocessor.FirstDate);
            Assert.Equal(new DateTime(2011, 01, 18, 19, 0, 0), _preprocessor.LastDate);
        }
    }
}