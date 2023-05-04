using System.Data;
using static TimeSeriesForecasting.DataPreprocessor;
using Record = TimeSeriesForecasting.IO.Record;

namespace TestTimeSeriesForecasting
{
    [Collection("Preprocessor collection")]
    public class TestPreprocessor
    {
        private readonly PreprocessorFixture _fixture;

        public TestPreprocessor(PreprocessorFixture fixture) => _fixture = fixture;

        [Fact]
        public void TestMinMaxNormalization()
        {
            // Normalize the data using Min-Max normalization.
            _fixture.Preprocessor.Normalization = NormalizationMethod.MIN_MAX_NORMALIZATION;
            DataTable table = _fixture.Preprocessor.GetTrainingSet();
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
                    Assert.True(Math.Abs(minValue) < _fixture.Tolerance);
                    // The maximum of any column must be 1.
                    Assert.True(Math.Abs(maxValue - 1) < _fixture.Tolerance);
                }
            }
        }

        [Fact]
        public void TestStandardization()
        {
            // Normalize the data using standardization.
            _fixture.Preprocessor.Normalization = NormalizationMethod.STANDARDIZATION;
            DataTable table = _fixture.Preprocessor.GetTrainingSet();
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
                    Assert.True(average < _fixture.Tolerance);
                    // The standard deviation of any column after standardization must be 1.
                    Assert.True(Math.Abs(stdev - 1) < _fixture.Tolerance);
                }
            }
        }

        [Fact]
        public void TestDateRange()
        {
            Assert.Equal(_fixture.FirstDatasetDate, _fixture.Preprocessor.FirstDate);
            Assert.Equal(_fixture.LastDatasetDate, _fixture.Preprocessor.LastDate);

            DateTime? firstDate = new DateTime(2010, 11, 25, 10, 0, 0);
            DateTime? lastDate = new DateTime(2010, 12, 31, 23, 0, 0);
            var preprocessor = _fixture.Preprocessor;
            preprocessor.DateRange = (firstDate, lastDate);
            Assert.Equal(firstDate, preprocessor.FirstDate);
            Assert.Equal(lastDate, preprocessor.LastDate);

            preprocessor.DateRange = (lastDate, null);
            Assert.Equal(lastDate, preprocessor.FirstDate);
            Assert.Equal(_fixture.LastDatasetDate, preprocessor.LastDate);

            DateTime? outOfBoundsFirstDate = new DateTime(2005, 1, 1, 0, 0, 0);
            DateTime? outOfBoundsLastDate = new DateTime(2020, 10, 5, 0, 0, 0);
            preprocessor.DateRange = (outOfBoundsFirstDate, outOfBoundsLastDate);
            Assert.Equal(_fixture.FirstDatasetDate, preprocessor.FirstDate);
            Assert.Equal(_fixture.LastDatasetDate, preprocessor.LastDate);
        }
    }
}
