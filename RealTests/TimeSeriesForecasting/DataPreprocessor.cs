using Google.Protobuf.WellKnownTypes;
using System.Data;
using System.Diagnostics;
using TimeSeriesForecasting.IO;

namespace TimeSeriesForecasting
{
    /// <summary>
    /// This class contains the logic that produces usable data from raw records.
    /// </summary>
    public class DataPreprocessor
    {
        public enum NormalizationMethod
        {
            NONE,
            MIN_MAX_NORMALIZATION,
            STANDARDIZATION
        }

        private const double SecondsInDay = 24 * 60 * 60;
        private const double SecondsInYear = 365.2425 * SecondsInDay;
        // This field is a workaround to allow tests on simpler datasets that don't have the expected features.
        private const bool FeatureEngineering = false;

        /*
         * Performing transformations on large datasets is really resource expensive,
         * so transformed data is cached in private fields instead of being re-calculated
         * every time it is needed.
         */
        private readonly DataTable _rawData;
        private readonly DataTable _processedData;
        private DataTable _normalizedData = new();
        private DataTable _dateLimitedData = new();
        private NormalizationMethod _normalization = NormalizationMethod.NONE;
        private DateTime _firstDate = DateTime.MinValue;
        private DateTime _lastDate = DateTime.MaxValue;

        public int TrainingSetPercentage { get; private set; }
        public int ValidationSetPercentage { get; private set; }
        public int TestSetPercentage { get; private set; }
        public NormalizationMethod Normalization
        {
            get => _normalization;
            set
            {
                _normalization = value;
                _normalizedData = ComputeNormalization();
                // DateRange must be updated so that the order of the operations in the pipeline is preserved.
                DateRange = Tuple.Create(new DateTime?(_firstDate), new DateTime?(_lastDate));
            }
        }
        public Tuple<DateTime?, DateTime?> DateRange
        {
            set
            {
                if (value.Item1.HasValue && value.Item2.HasValue && value.Item1.Value >= value.Item2.Value)
                {
                    throw new ArgumentException("The first item must be earlier than the second item.");
                }
                DateTime datasetMinDate = (DateTime)_normalizedData.Rows[0][Record.Index];
                DateTime datasetMaxDate = (DateTime)_normalizedData.Rows[^1][Record.Index];
                /*
                 * If the first item in the given tuple is earlier than the first date in the dataset, then the first date
                 * is set to the first date in the dataset, so it is basically a reset. The same is true for the last date.
                 */
                _firstDate = value.Item1.HasValue ?
                                new DateTime(Math.Max(value.Item1.Value.Ticks, datasetMinDate.Ticks)) : datasetMinDate;
                _lastDate = value.Item2.HasValue ?
                                new DateTime(Math.Min(value.Item2.Value.Ticks, datasetMaxDate.Ticks)) : datasetMaxDate;
                _dateLimitedData = LimitDateRange();
            }
        }
        public DateTime FirstDate { get => _firstDate; }
        public DateTime LastDate { get => _lastDate; }

        /// <summary>
        /// Creates a new instance of DataPreprocessor to operate on the given <see cref="IList{Record}"/>.
        /// All processing parameters, such as the normalization method and the proportion of data to be
        /// included in the training, validation and test sets are assigned to the default values.
        /// </summary>
        /// <param name="records">A list of <see cref="Record"/>s that represent phenomenon observations.</param>
        public DataPreprocessor(IList<Record> records) : 
            this(records, new Tuple<int, int, int>(70, 20, 10), NormalizationMethod.NONE, Tuple.Create<DateTime?, DateTime?>(null, null)) 
        {
        }

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
                                NormalizationMethod normalization, Tuple<DateTime?, DateTime?> range)
        {
            TrainingSetPercentage = splitter.Item1;
            ValidationSetPercentage = splitter.Item2;
            TestSetPercentage = splitter.Item3;
            // Remove duplicate elements according to the primary key (timestamp).
            var uniqueRecords = records.DistinctBy(r => r.TimeStamp).ToList();
            // Store data in table format.
            _rawData = new DataTable();
            _rawData.Columns.Add(Record.Index, typeof(DateTime));
            records.First().Features.Keys.ToList().ForEach(key => _rawData.Columns.Add(key, typeof(double)));
            _rawData.PrimaryKey = new DataColumn[] { _rawData.Columns[Record.Index]! };
            foreach (var record in uniqueRecords)
            {
                var row = _rawData.NewRow();
                row[Record.Index] = record.TimeStamp;
                foreach ((string feature, double val) in record.Features)
                {
                    row[feature] = val;
                }
                _rawData.Rows.Add(row);
            }
            // Sort data for timestamp, it is needed in debugging below
            _rawData.DefaultView.Sort = $"{Record.Index} ASC";
            _rawData = _rawData.DefaultView.ToTable();
            int daysBetweenFirstAndLastDate = ((DateTime)_rawData.Rows[^1][Record.Index])
                                                .Subtract((DateTime)_rawData.Rows[0][Record.Index]).Days;
            /* 
             * This is the preprocessing PIPELINE: first data is processed, i.e. clear measurement errors are cleaned up and
             * new features are engineered, then data is normalized using the given normalization method and finally records
             * are removed if their timestamp is not inside the given range. After these three lines of code are executed,
             * data is ready to be acquired by the client through the following Get___Set() methods.
             */
            _processedData = ProcessData();
            Normalization = normalization;
            DateRange = range;
            /*
             * Debugging of the pipeline
             */
            // Only one sixth of all the values are in the new table, as observations taken at fractions of hours have been removed.
            Trace.Assert(Math.Abs((double)_processedData.Rows.Count / _rawData.Rows.Count - 0.166666) < 10e-2);
            // Normalization should neither add nor remove records from the table.
            Trace.Assert(_normalizedData.Rows.Count == _processedData.Rows.Count);
            // Number of rows of new table : Number of rows of normalized table =
            // Distance between client range dates : distance between dataset extreme dates
            int newDaysBetweenFirstAndLastDate = ((DateTime)_dateLimitedData.Rows[^1][Record.Index])
                                                            .Subtract((DateTime)_dateLimitedData.Rows[0][Record.Index]).Days;
            Trace.Assert(Math.Abs((double)_normalizedData.Rows.Count / _dateLimitedData.Rows.Count -
                                    (double)daysBetweenFirstAndLastDate / newDaysBetweenFirstAndLastDate) < 10e-2);
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

        // This method is only called ONCE, from inside the constructor, to skim data that might then be modified further.
        private DataTable ProcessData()
        {
            // Raw data cleanup, i.e. removal of clear measurement errors.
            _rawData.Rows.Cast<DataRow>().ToList().ForEach(row =>
            {
                _rawData.Columns.Cast<DataColumn>()
                                .Where(col => Record.GetUnitOfMeasureFromFeatureName(col.ColumnName) != null)
                                .ToList()
                                .ForEach(col =>
                {
                    string colName = col.ColumnName;
                    string unit = Record.GetUnitOfMeasureFromFeatureName(colName)!;
                    double value = (double)row[colName];
                    (double minValue, double maxValue) = Record.ValueRanges[unit];
                    // Replace all values of a row that are outside the allowed values boundaries.
                    row[colName] = Math.Max(minValue, Math.Min(maxValue, value));
                });
            });
            // Processed data uses only hourly values.
            DataTable processedData = _rawData.AsEnumerable().Where(r => ((DateTime?)r.ItemArray[0])?.Minute == 0).CopyToDataTable();
            // Feature engineering, performed only if not testing.
            if (FeatureEngineering)
            {
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
            if (Normalization == NormalizationMethod.NONE)
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
                if (Normalization == NormalizationMethod.MIN_MAX_NORMALIZATION)
                {
                    var normalizedData = _processedData.Copy();
                    foreach (DataColumn col in normalizedData.Columns)
                    {
                        var name = col.ColumnName;
                        if (name != Record.Index)
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
                            if (col.ColumnName != Record.Index)
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
                        if (name != Record.Index)
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
                            if (col.ColumnName != Record.Index)
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
            return _normalizedData
                .AsEnumerable()
                .Where(dr => dr.Field<DateTime>(Record.Index) >= FirstDate && dr.Field<DateTime>(Record.Index) <= LastDate)
                .CopyToDataTable();
        }
    }
}
