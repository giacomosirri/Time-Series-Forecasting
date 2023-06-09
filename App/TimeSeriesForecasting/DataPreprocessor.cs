﻿using MoreLinq;
using System.Data;
using System.Diagnostics;
using TimeSeriesForecasting.IO;

namespace TimeSeriesForecasting
{
    /// <summary>
    /// This class contains the logic that produces usable data from raw records.
    /// </summary>
    public class DataPreprocessor : IDisposable
    {
        public enum NormalizationMethod
        {
            NONE,
            MIN_MAX_NORMALIZATION,
            STANDARDIZATION
        }

        /*
         * Performing transformations on large datasets is very resource expensive,
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
                DateRange = (_firstDate, _lastDate);
            }
        }

        public (DateTime? firstValidDate, DateTime? lastValidDate) DateRange
        {
            set
            {
                if (value.firstValidDate.HasValue && 
                    value.lastValidDate.HasValue &&
                    value.firstValidDate.Value >= value.lastValidDate.Value)
                {
                    throw new ArgumentException("The first item must be earlier than the second item.");
                }
                DateTime datasetMinDate = (DateTime)_normalizedData.Rows[0][Record.Index];
                DateTime datasetMaxDate = (DateTime)_normalizedData.Rows[^1][Record.Index];
                /*
                 * If the first item in the given tuple is earlier than the first date in the dataset, then the first date
                 * is set to the first date in the dataset, so it is basically a reset. The same is true for the last date.
                 */
                _firstDate = value.firstValidDate.HasValue ?
                                new DateTime(Math.Max(value.firstValidDate.Value.Ticks, datasetMinDate.Ticks)) : datasetMinDate;
                _lastDate = value.lastValidDate.HasValue ?
                                new DateTime(Math.Min(value.lastValidDate.Value.Ticks, datasetMaxDate.Ticks)) : datasetMaxDate;
                _dateLimitedData = LimitDateRange();
            }
        }

        public DateTime FirstDate { get => _firstDate; }

        public DateTime LastDate { get => _lastDate; }

        internal DataPreprocessor(IList<Record> records, (int training, int validation, int test) splits,
                                NormalizationMethod normalization, (DateTime? firstValidDate, DateTime? lastValidDate) range)
        {
            TrainingSetPercentage = splits.training;
            ValidationSetPercentage = splits.validation;
            TestSetPercentage = splits.test;
            // Remove duplicate elements according to the primary key (timestamp).
            var uniqueRecords = MoreEnumerable.DistinctBy(records, r => r.TimeStamp).ToList();
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
            // Sort data for timestamp, it is needed in debugging below.
            _rawData.DefaultView.Sort = $"{Record.Index} ASC";
            _rawData = _rawData.DefaultView.ToTable();
            int daysBetweenFirstAndLastDate = ((DateTime)_rawData.Rows[^1][Record.Index])
                                                .Subtract((DateTime)_rawData.Rows[0][Record.Index]).Days;
            /* 
             * This is the preprocessing PIPELINE: first data is processed, i.e. clear measurement errors are cleaned up and
             * new features are engineered, then data is normalized using the given normalization method and finally records
             * are removed if their timestamp is not inside the given range. After these three lines of code are executed,
             * data is ready to be acquired by the client through the Get___Set() methods below.
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

        public DataTable GetTrainingSet() => GetSet(TrainingSetPercentage, 0);

        public DataTable GetValidationSet() => GetSet(ValidationSetPercentage, TrainingSetPercentage);

        public DataTable GetTestSet() => GetSet(TestSetPercentage, TrainingSetPercentage + ValidationSetPercentage);

        private DataTable GetSet(int takePercentage, int skipPercentage)
        {
            int takeRows = (int)Math.Round(_dateLimitedData.Rows.Count * takePercentage / 100.0);
            int skipRows = (int)Math.Round(_dateLimitedData.Rows.Count * skipPercentage / 100.0);
            DataTable result = _dateLimitedData.Clone();
            for (int i = skipRows; i < skipRows + takeRows; i++)
            {
                result.ImportRow(_dateLimitedData.Rows[i]);
            }
            return result;
        }

        // This method is only called ONCE, from inside the constructor, to skim data that might then be modified further.
        private DataTable ProcessData()
        {
            // Raw data cleanup, i.e. removal of clear measurement errors.
            _rawData.Rows.Cast<DataRow>().ToList().ForEach(row => _rawData.Columns.Cast<DataColumn>()
                .Select(col => col.ColumnName)
                .Where(colName => Record.ValueRanges.ContainsKey(colName))
                .ToList()
                .ForEach(colName =>
                {
                    // Current value of the field.
                    double value = (double)row[colName];
                    double minValue = Record.ValueRanges[colName].minValue ?? double.MinValue;
                    double maxValue = Record.ValueRanges[colName].maxValue ?? double.MaxValue;
                    // Replace all values of a row that are outside the allowed values boundaries.
                    row[colName] = Math.Max(minValue, Math.Min(maxValue, value));
                })
            );
            // Processed data uses only hourly values.
            return _rawData.AsEnumerable().Where(r => ((DateTime?)r.ItemArray[0])?.Minute == 0).CopyToDataTable();
        }

        /*
         * This method returns a table containing ALL the rows of the main table, 
         * normalized according to the Normalization property.
         * The behavior is to normalize using only the training set time steps,
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
                /*
                 * This dictionary associates the name of a column to its minimum and maximum value 
                 * or to its average and standard deviation depending on the normalization method used.
                 */
                IDictionary<string, Tuple<double, double>> parameters = new Dictionary<string, Tuple<double, double>>();
                // Create training set from the full set of data to calculate the normalization table.
                var normalizationTable = _processedData.Clone();
                int rows = (int)Math.Round(_processedData.Rows.Count * TrainingSetPercentage / 100.0);
                for (int i = 0; i < rows; i++)
                {
                    normalizationTable.ImportRow(_processedData.Rows[i]);
                }
                if (Normalization == NormalizationMethod.MIN_MAX_NORMALIZATION)
                {
                    normalizationTable.Columns
                        .Cast<DataColumn>()
                        .Select(col => col.ColumnName)
                        .Where(cn => cn != Record.Index)
                        .ForEach(cn =>
                        {
                            double min = normalizationTable.AsEnumerable().Min(r => r.Field<double>(cn));
                            double max = normalizationTable.AsEnumerable().Max(r => r.Field<double>(cn));
                            parameters.Add(cn, new Tuple<double, double>(min, max));
                        });
                    DataTable normalizedData = _processedData.Copy();
                    normalizedData.AsEnumerable().ForEach(row => normalizedData.Columns
                                                            .Cast<DataColumn>()
                                                            .Select(col => col.ColumnName)
                                                            .Where(cn => cn != Record.Index)
                                                            .ForEach(cn => row[cn] = ((double)row[cn] - parameters[cn].Item1) /
                                                                (parameters[cn].Item2 - parameters[cn].Item1)));
                    return normalizedData;
                }
                else
                {
                    IList<Tuple<string, double>> averages = normalizationTable.Columns
                        .Cast<DataColumn>()
                        .Select(col => col.ColumnName)
                        .Where(cn => cn != Record.Index)
                        .Select(cn => Tuple.Create(cn, normalizationTable.AsEnumerable().Average(r => r.Field<double>(cn))))
                        .ToList();
                    IList<Tuple<string, double>> standardDeviations = normalizationTable.Columns
                        .Cast<DataColumn>()
                        .Select(col => col.ColumnName)
                        .Where(cn => cn != Record.Index)
                        .Select(cn => Tuple.Create(cn, CalculateStDev(normalizationTable.AsEnumerable().Select(row => row.Field<double>(cn)))))
                        .ToList();
                    parameters = averages
                        .Zip(standardDeviations, (avg, stdev) => new { Key = avg.Item1, Value = Tuple.Create(avg.Item2, stdev.Item2) })
                        .ToDictionary(x => x.Key, x => x.Value);
                    DataTable normalizedData = _processedData.Copy();
                    normalizedData.AsEnumerable().ForEach(row => normalizedData.Columns
                                                            .Cast<DataColumn>()
                                                            .Select(col => col.ColumnName)
                                                            .Where(cn => cn != Record.Index)
                                                            .ForEach(cn => row[cn] = ((double)row[cn] - parameters[cn].Item1) / 
                                                                parameters[cn].Item2));
                    return normalizedData;
                }
            }
        }

        private static double CalculateStDev(IEnumerable<double> values)
        {
            double ret = 0;
            if (values.Any())
            {
                double avg = values.Average();
                double sum = values.Sum(d => Math.Pow(d - avg, 2));
                ret = Math.Sqrt(sum / values.Count());
            }
            return ret;
        }

        private DataTable LimitDateRange()
        {
            return _normalizedData
                .AsEnumerable()
                .Where(dr => dr.Field<DateTime>(Record.Index) >= FirstDate && dr.Field<DateTime>(Record.Index) <= LastDate)
                .CopyToDataTable();
        }

        public void Dispose()
        {
            _rawData.Dispose();
            _processedData.Dispose();
            _normalizedData.Dispose();
            _dateLimitedData.Dispose();
        }
    }
}
