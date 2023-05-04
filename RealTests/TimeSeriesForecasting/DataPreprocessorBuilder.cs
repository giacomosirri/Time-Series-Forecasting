using TimeSeriesForecasting.IO;

namespace TimeSeriesForecasting
{
    /// <summary>
    /// Creates a <see cref="DataPreprocessor"/> object through the Builder design pattern.
    /// </summary>
    internal class DataPreprocessorBuilder : IDataPreprocessorBuilder
    {
        private (DateTime? firstDate, DateTime? lastDate) _range = (null, null);
        private DataPreprocessor.NormalizationMethod _normalization = DataPreprocessor.NormalizationMethod.NONE;
        private (int training, int validation, int test) _splits = (70, 20, 10);

        /// <inheritdoc/>
        IDataPreprocessorBuilder IDataPreprocessorBuilder.AddDateRange((DateTime? firstDate, DateTime? lastDate) range)
        {
            _range = range;
            return this;
        }

        /// <inheritdoc/>
        IDataPreprocessorBuilder IDataPreprocessorBuilder.Normalize(DataPreprocessor.NormalizationMethod normalization)
        {
            _normalization = normalization;
            return this;
        }

        /// <inheritdoc/>
        IDataPreprocessorBuilder IDataPreprocessorBuilder.Split((int training, int validation, int test) splits)
        {
            _splits = splits;
            return this;
        }

        /// <inheritdoc/>
        DataPreprocessor IDataPreprocessorBuilder.Build(IList<Record> records)
        {
            return new DataPreprocessor(records, _splits, _normalization, _range);
        }
    }
}
