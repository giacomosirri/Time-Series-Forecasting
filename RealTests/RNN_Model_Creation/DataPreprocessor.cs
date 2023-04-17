using Google.Protobuf.WellKnownTypes;
using System.Data;

namespace RNN_Model_Creation
{
    /// <summary>
    /// This class contains the logic that produces usable data from raw records.
    /// </summary>
    internal class DataPreprocessor
    {
        private static readonly double SecondsInDay = 24 * 60 * 60;
        private static readonly double SecondsInYear = 365.2425 * SecondsInDay;

        private readonly DataTable _rawData;
        private readonly DataTable _processedData;

        public int TrainingSetPercentage { get; private set; }
        public int ValidationSetPercentage { get; private set; }
        public int TestSetPercentage { get; private set; }
        public string Normalization { get; set; }
        
        public DataPreprocessor(IList<Record> records) : this(records, new Tuple<int, int, int>(70,20,10), "None") { }

        public DataPreprocessor(IList<Record> records, Tuple<int, int, int> splitter, string normalization)
        {
            TrainingSetPercentage = splitter.Item1;
            ValidationSetPercentage = splitter.Item2;
            TestSetPercentage = splitter.Item3;
            Normalization = normalization;
            // remove duplicate elements according to the primary key (date)
            var uniqueRecords = RemoveDuplicateRecords(records);
            // store data in table format
            _rawData = new DataTable();
            _rawData.Columns.Add("Date Time", typeof(DateTime));
            records.First().Features.Keys.ToList().ForEach(key => _rawData.Columns.Add(key, typeof(double)));
            _rawData.PrimaryKey = new DataColumn[] { _rawData.Columns["Date Time"]! };
            foreach (var record in uniqueRecords)
            {
                var row = _rawData.NewRow();
                row["Date Time"] = record.TimeStamp;
                foreach ((string feature, double val) in record.Features)
                {
                    row[feature] = val;
                }
                _rawData.Rows.Add(row);
            }
            _processedData = ProcessData();
        }

        public DataTable GetTrainingSet()
        {
            return ComputeNormalization().AsEnumerable().Take(TrainingSetPercentage).CopyToDataTable();
        }

        public DataTable GetValidationSet()
        {
            return ComputeNormalization().AsEnumerable().Take(ValidationSetPercentage).CopyToDataTable();
        }

        public DataTable GetTestSet()
        {
            return ComputeNormalization().AsEnumerable().Take(TestSetPercentage).CopyToDataTable();
        }

        private DataTable ProcessData()
        {
            // RAW DATA CLEANUP
            var processedData = _rawData.Clone();
            foreach (DataRow row in _rawData.Rows)
            {
                var date = (DateTime)row["Date Time"];
                // let's deal with hourly values only
                if (date.Minute == 0)
                {
                    // replace all values of a row that are outside the allowed values boundaries...
                    foreach (DataColumn col in _rawData.Columns)
                    {
                        string colName = col.ColumnName;
                        string? unit = Record.GetUnitOfMeasureFromFeatureName(colName);
                        if (unit != null)
                        {
                            if (Record.MinMaxPossibleValues[unit].Item1 > ((double)row[colName]))
                            {
                                row[colName] = Record.MinMaxPossibleValues[unit].Item1;
                            }
                            if (Record.MinMaxPossibleValues[unit].Item2 < ((double)row[colName]))
                            {
                                row[colName] = Record.MinMaxPossibleValues[unit].Item2;
                            }
                        }
                    }
                    // .. then add the row to the new processed data table
                    processedData.ImportRow(_rawData.Rows.Find(date));
                }
            }
            // FEATURE ENGINEERING
            processedData.Columns.Add("wx (m/s)", typeof(double));
            processedData.Columns.Add("wy (m/s)", typeof(double));
            processedData.Columns.Add("max. wx (m/s)", typeof(double));
            processedData.Columns.Add("max. wy (m/s)", typeof(double));
            processedData.Columns.Add("day sin", typeof(double));
            processedData.Columns.Add("day cos", typeof(double));
            processedData.Columns.Add("year sin", typeof(double));
            processedData.Columns.Add("year cos", typeof(double));
            foreach (DataRow row in processedData.Rows) 
            {
                double windVelocity = (double)row["wv (m/s)"];
                double maximumWindVelocity = (double)row["max. wv (m/s)"];
                double degreeInRadiants = (double) row["wd (deg)"] * Math.PI / 180;
                row["wx (m/s)"] = windVelocity * Math.Cos(degreeInRadiants);
                row["wy (m/s)"] = windVelocity * Math.Sin(degreeInRadiants);
                row["max. wx (m/s)"] = maximumWindVelocity * Math.Cos(degreeInRadiants);
                row["max. wy (m/s)"] = maximumWindVelocity * Math.Sin(degreeInRadiants);
                double secondsSinceEpoch = ((DateTime)row["Date Time"]).ToUniversalTime().ToTimestamp().Seconds;
                row["day sin"] = Math.Sin(secondsSinceEpoch * (2 * Math.PI / SecondsInDay));
                row["day cos"] = Math.Cos(secondsSinceEpoch * (2 * Math.PI / SecondsInDay));
                row["year sin"] = Math.Sin(secondsSinceEpoch * (2 * Math.PI / SecondsInYear));
                row["year cos"] = Math.Cos(secondsSinceEpoch * (2 * Math.PI / SecondsInYear));
            }
            return processedData;
        }

        private static IList<Record> RemoveDuplicateRecords(IList<Record> records) => records.DistinctBy(r => r.TimeStamp).ToList();

        private DataTable ComputeNormalization()
        {
            if (Normalization == "Normalization") 
            {
                return new DataTable();
            }
            else if (Normalization == "Standardization")
            {
                return new DataTable();
            }
            return _processedData;
        }
    }
}
