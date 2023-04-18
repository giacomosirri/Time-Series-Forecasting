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

        /* 
         * Performing transformations on large datasets is really resource expensive,
         * so transformed data is cached in private fields instead of being re-calculated
         * every time it is needed.
         */
        private readonly DataTable _rawData;
        private readonly DataTable _processedData;
        private DataTable _normalizedData = new();
        private string _normalization = "None";

        public int TrainingSetPercentage { get; private set; }
        public int ValidationSetPercentage { get; private set; }
        public int TestSetPercentage { get; private set; }
        public string Normalization
        {
            get => _normalization;
            set
            {
                if (value == "Normalization" || value == "Standardization" || value == "None")
                {
                    _normalization = value;
                    _normalizedData = ComputeNormalization();
                }
                else
                {
                    _normalization = "None";
                    _normalizedData = _processedData;
                    throw new ArgumentException("The normalization method provided is not supported. " +
                        "Dataset is back to being not-normalized.");
                }
            }
        }
        
        public DataPreprocessor(IList<Record> records) : this(records, new Tuple<int, int, int>(70,20,10), "None") { }

        public DataPreprocessor(IList<Record> records, Tuple<int, int, int> splitter, string normalization)
        {
            TrainingSetPercentage = splitter.Item1;
            ValidationSetPercentage = splitter.Item2;
            TestSetPercentage = splitter.Item3;
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
            Normalization = normalization;
        }

        public DataTable GetTrainingSet() => GetSet(TrainingSetPercentage);

        public DataTable GetValidationSet() => GetSet(ValidationSetPercentage);

        public DataTable GetTestSet() => GetSet(TestSetPercentage);

        private DataTable GetSet(int percentage)
        {
            var newSet = _normalizedData.Clone();
            int rows = (int)Math.Round(_normalizedData.Rows.Count * percentage / 100.0);
            for (int i = 0; i < rows; i++)
            {
                newSet.ImportRow(_normalizedData.Rows[i]);
            }
            return newSet;
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

        /*
         * This method returns a table containing ALL the rows of the main table, 
         * normalized according to the Normalization property.
         * The behavior is to normalize using only the training set observations,
         * i.e. the first 70% of the rows, since it prevents information leakage
         * from the validation and test set to the training set.
         */
        private DataTable ComputeNormalization()
        {
            if (Normalization == "None")
            {
                return _processedData;
            }
            else
            {
                // create training set from the full set of data to calculate the normalization table
                var normalizationTable = _processedData.Clone();
                int rows = (int)Math.Round(_processedData.Rows.Count * TrainingSetPercentage / 100.0);
                for (int i = 0; i < rows; i++)
                {
                    normalizationTable.ImportRow(_processedData.Rows[i]);
                }
                IDictionary<string, Tuple<double, double>> values = new Dictionary<string, Tuple<double, double>>();
                if (Normalization == "Normalization")
                {
                    var normalizedData = _processedData.Copy();
                    foreach (DataColumn col in normalizedData.Columns)
                    {
                        var name = col.ColumnName;
                        var min = Convert.ToDouble(normalizationTable.Compute($"Min([{col.ColumnName}])", ""));
                        var max = Convert.ToDouble(normalizationTable.Compute($"Max([{col.ColumnName}])", ""));
                        values.Add(name, new Tuple<double, double>(min, max));
                    }
                    foreach (DataRow row in normalizedData.Rows)
                    {
                        foreach (DataColumn col in normalizedData.Columns)
                        {
                            var min = values[col.ColumnName].Item1;
                            var max = values[col.ColumnName].Item2;
                            row[col] = ((double)row[col] - min) / (max - min);
                        }
                    }
                    return normalizedData;
                }
                else
                {
                    var standardizedData = _processedData.Copy();
                    foreach (DataColumn col in standardizedData.Columns)
                    {
                        var name = col.ColumnName;
                        if (name != "Date Time")
                        {
                            var avg = Convert.ToDouble(normalizationTable.Compute($"Avg([{col.ColumnName}])", ""));
                            var std = Math.Sqrt(-Convert.ToDouble(normalizationTable.Compute($"Var([{col.ColumnName}])", "")));
                            values.Add(name, new Tuple<double, double>(avg, std));
                        }
                    }
                    //Console.WriteLine(string.Join(", ", values.Values));
                    foreach (DataRow row in standardizedData.Rows)
                    {
                        foreach (DataColumn col in standardizedData.Columns)
                        {
                            if (col.ColumnName != "Date Time")
                            {
                                var avg = values[col.ColumnName].Item1;
                                var std = values[col.ColumnName].Item2;
                                row[col] = ((double)row[col] - avg) / std;
                            }
                        }
                    }
                    return standardizedData;
                }
            }
        }
    }
}
