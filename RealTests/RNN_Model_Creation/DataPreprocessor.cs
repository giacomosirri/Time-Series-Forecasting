using System.Data;

namespace RNN_Model_Creation
{
    /// <summary>
    /// This class contains the logic that produces usable data from raw records.
    /// </summary>
    internal class DataPreprocessor
    {
        private readonly DataTable _data;

        public int TrainingSetPercentage { get; private set; }
        public int ValidationSetPercentage { get; private set; }
        public int TestSetPercentage { get; private set; }
        
        public DataPreprocessor(IList<Record> records) : this(records, new Tuple<int, int, int>(70,20,10)) { }

        public DataPreprocessor(IList<Record> records, Tuple<int, int, int> splitter)
        {
            TrainingSetPercentage = splitter.Item1;
            ValidationSetPercentage = splitter.Item2;
            TestSetPercentage = splitter.Item3;
            // remove duplicate elements according to the primary key (date)
            var uniqueRecords = RemoveDuplicateRecords(records);
            // store data in table format
            _data = new DataTable();
            _data.Columns.Add("Date Time", typeof(DateTime));
            records.First().Features.Keys.ToList().ForEach(key => _data.Columns.Add(key, typeof(double)));
            _data.PrimaryKey = new DataColumn[] { _data.Columns["Date Time"]! };
            foreach (var record in uniqueRecords)
            {
                var row = _data.NewRow();
                row["Date Time"] = record.TimeStamp;
                foreach ((string feat, double val) in record.Features)
                {
                    row[feat] = val;
                }
                _data.Rows.Add(row);
            }
            ProcessData();
        }

        public DataTable GetProcessedTrainingSet() => _data.AsEnumerable().Take(TrainingSetPercentage).CopyToDataTable();

        public DataTable GetProcessedValidationSet() => _data.AsEnumerable().Take(ValidationSetPercentage).CopyToDataTable();

        public DataTable GetProcessedTestSet() => _data.AsEnumerable().Take(TestSetPercentage).CopyToDataTable();

        private static IList<Record> RemoveDuplicateRecords(IList<Record> records) => records.DistinctBy(r => r.TimeStamp).ToList();

        private void ProcessData()
        {
        }
    }
}
