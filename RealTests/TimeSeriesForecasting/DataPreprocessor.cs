using Google.Protobuf.WellKnownTypes;
using System.Data;
using TimeSeriesForecasting.IO;

namespace TimeSeriesForecasting
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
        private DataTable _dateLimitedData = new();
        private string _normalization = "None";
        private DateTime? _firstDate;
        private DateTime? _lastDate;

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
                else throw new ArgumentException("The normalization method provided is not supported.");
            }
        }
        public Tuple<DateTime?, DateTime?> DateRange
        {
            set
            {
                _firstDate = value.Item1;
                _lastDate = value.Item2;
                _dateLimitedData = LimitDateRange();
            }
        }

        /// <summary>
        /// Creates a new instance of DataPreprocessor to operate on the given <see cref="IList{Record}"/>.
        /// All processing parameters, such as the normalization method and the proportion of data to be
        /// included in the training, validation and test sets are assigned to the default values.
        /// </summary>
        /// <param name="records">A list of <see cref="Record"/>s that represent phenomenon observations.</param>
        public DataPreprocessor(IList<Record> records) : this(records, new Tuple<int, int, int>(70, 20, 10), 
                                                                "None", Tuple.Create<DateTime?, DateTime?>(null, null)) {}

        /// <summary>
        /// Creates a new instance of DataPreprocessor, with custom parameters to suit the needs of the client.
        /// </summary>
        /// <param name="records">A list of <see cref="Record"/>s that represent phenomenon observations.</param>
        /// <param name="splitter">A <see cref="Tuple"/> with the percentages of values to be included in the 
        /// training, validation and test set respectively.</param>
        /// <param name="normalization">The normalization method. There are three allowed values: "Normalization" for
        /// Min-Max Normalization, "Standardization" for Z-Score and "None" for no normalization.</param>
        /// <param name="range">A <see cref="Tuple"/> that contains the first and last date to be included in the
        /// processed data. Can be useful to speed up processing if the dataset contains dozens of thousands of 
        /// observations or even more.</param>
        public DataPreprocessor(IList<Record> records, Tuple<int, int, int> splitter, 
                                string normalization, Tuple<DateTime?, DateTime?> range)
        {
            TrainingSetPercentage = splitter.Item1;
            ValidationSetPercentage = splitter.Item2;
            TestSetPercentage = splitter.Item3;
            // Remove duplicate elements according to the primary key (timestamp).
            var uniqueRecords = records.DistinctBy(r => r.TimeStamp).ToList();
            // Store data in table format.
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
            /* 
             * This is the preprocessing PIPELINE: first data is processed, i.e. clear measurement errors are cleaned up and
             * new features are engineered, then data is normalized using the given normalization method and finally records
             * are removed if their timestamp is not inside the given range. After these three lines of code are executed,
             * data is ready to be acquired by the client through the following Get___Set() methods.
             */
            _processedData = ProcessData();
            Normalization = normalization;
            DateRange = range;
        }

        public DataTable GetTrainingSet() => GetSet(TrainingSetPercentage);

        public DataTable GetValidationSet() => GetSet(ValidationSetPercentage);

        public DataTable GetTestSet() => GetSet(TestSetPercentage);

        private DataTable GetSet(int percentage)
        {
            DataTable result = _dateLimitedData.Clone();
            int rows = (int)Math.Round(_dateLimitedData.Rows.Count * percentage / 100.0);
            for (int i = 0; i < rows; i++)
            {
                result.ImportRow(_dateLimitedData.Rows[i]);
            }
            return result;
        }

        public DateTime? GetFirstValideDate() => _firstDate;

        public DateTime? GetLastValidDate() => _lastDate;

        private DataTable ProcessData()
        {
            // RAW DATA CLEANUP
            var processedData = _rawData.Clone();
            foreach (DataRow row in _rawData.Rows)
            {
                var date = (DateTime)row["Date Time"];
                // Let's deal with hourly values only.
                if (date.Minute == 0)
                {
                    // Replace all values of a row that are outside the allowed values boundaries...
                    foreach (DataColumn col in _rawData.Columns)
                    {
                        string colName = col.ColumnName;
                        string? unit = Record.GetUnitOfMeasureFromFeatureName(colName);
                        if (unit != null)
                        {
                            if (Record.ValueRanges[unit].Item1 > (double)row[colName])
                            {
                                row[colName] = Record.ValueRanges[unit].Item1;
                            }
                            if (Record.ValueRanges[unit].Item2 < (double)row[colName])
                            {
                                row[colName] = Record.ValueRanges[unit].Item2;
                            }
                        }
                    }
                    // ... Then add the row to the new processed data table.
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
                double degreeInRadiants = (double)row["wd (deg)"] * Math.PI / 180;
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
                // Create training set from the full set of data to calculate the normalization table.
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
                        if (name != "Date Time")
                        {
                            var min = Convert.ToDouble(normalizationTable.Compute($"Min([{col.ColumnName}])", ""));
                            var max = Convert.ToDouble(normalizationTable.Compute($"Max([{col.ColumnName}])", ""));
                            values.Add(name, new Tuple<double, double>(min, max));
                        }
                    }
                    foreach (DataRow row in normalizedData.Rows)
                    {
                        foreach (DataColumn col in normalizedData.Columns)
                        {
                            if (col.ColumnName != "Date Time")
                            {
                                var min = values[col.ColumnName].Item1;
                                var max = values[col.ColumnName].Item2;
                                row[col] = ((double)row[col] - min) / (max - min);
                            }
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

        private DataTable LimitDateRange()
        {
            DataTable newSet = _normalizedData.Copy();
            Console.WriteLine(newSet.Rows.Count);
            if (_firstDate.HasValue)
            {
                newSet = newSet.AsEnumerable()
                               .Where(dr => dr.Field<DateTime>(newSet.Columns["Date Time"]!) >= _firstDate.Value)
                               .CopyToDataTable();
            }
            Console.WriteLine(newSet.Rows.Count);
            if (_lastDate.HasValue)
            {
                newSet = newSet.AsEnumerable()
                               .Where(dr => dr.Field<DateTime>(newSet.Columns["Date Time"]!) <= _lastDate.Value)
                               .CopyToDataTable();
            }
            Console.WriteLine(newSet.Rows.Count);
            return newSet;
        }
    }
}
