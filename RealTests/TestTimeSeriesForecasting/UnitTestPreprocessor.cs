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
            DateTime? startDate = new DateTime(10, 10, 2010);
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
            _preprocessor.Normalization = "Normalization";
            DataTable trainingSet = _preprocessor.GetTrainingSet();
            for (int i = 0; i<trainingSet.Columns.Count; i++) 
            {
                DataColumn col = trainingSet.Columns[i];
                if (col.ColumnName != "Date Time")
                {
                    double min = Convert.ToDouble(trainingSet.Compute($"Min([{col.ColumnName}])", ""));
                    double max = Convert.ToDouble(trainingSet.Compute($"Min([{col.ColumnName}])", ""));
                    Assert.True(Math.Abs(min - 0.0) < Tolerance);
                    Assert.True(Math.Abs(max - 1.0) < Tolerance);
                }
            }
        }
    }
}