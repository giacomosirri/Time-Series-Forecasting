using TimeSeriesForecasting.IO;

namespace TimeSeriesForecasting
{
    /// <summary>
    /// Creates a <see cref="DataPreprocessor"/> object through the Builder design pattern.
    /// Data preprocessing is the set of operations performed on raw data to make it suitable
    /// for machine learning and deep learning models. There are a lot of different operations
    /// that can be grouped under this term. This builder allows to perform some common data
    /// preprocessing operations before the <see cref="DataPreprocessor"/> instance is even 
    /// created.
    /// </summary>
    internal class DataPreprocessorBuilder
    {
        private (DateTime? firstDate, DateTime? lastDate) _range;
        private DataPreprocessor.NormalizationMethod _normalization;
        private (int training, int validation, int test) _splits;

        /// <summary>
        /// Creates a new instance of <see cref="DataPreprocessorBuilder"/>.
        /// </summary>
        internal DataPreprocessorBuilder()
        {
            _range = (null, null);
            _normalization = DataPreprocessor.NormalizationMethod.NONE;
            _splits = (70, 20, 10);
        }

        /// <summary>
        /// Splits the data into training, validation and test sets.
        /// </summary>
        /// <param name="splits">A <see cref="Tuple"/> with the percentages of values to be included in the 
        /// training, validation and test set respectively.</param>
        /// <returns>This instance of <see cref="DataPreprocessorBuilder"/>.</returns>
        internal DataPreprocessorBuilder Split((int training, int validation, int test) splits)
        {
            _splits = splits;
            return this;
        }

        /// <summary>
        /// Normalize the data.
        /// </summary>
        /// <param name="normalization">The normalization method.</param>
        /// <returns>This instance of <see cref="DataPreprocessorBuilder"/>.</returns>
        internal DataPreprocessorBuilder Normalize(DataPreprocessor.NormalizationMethod normalization)
        {
            _normalization = normalization;
            return this;
        }

        /// <summary>
        /// Clips the data to the given date boundaries.
        /// </summary>
        /// <param name="range">A <see cref="Tuple"/> that contains the first and last date to be included in the
        /// processed data. It might be useful to speed up processing if the dataset contains dozens of thousands of 
        /// observations or even more.</param>
        /// <returns>This instance of <see cref="DataPreprocessorBuilder"/>.</returns>
        internal DataPreprocessorBuilder AddDateRange((DateTime? firstDate, DateTime? lastDate) range)
        {
            _range = range;
            return this;
        }

        /// <summary>
        /// Creates a new instance of DataPreprocessor, with the custom parameters previously specified.
        /// </summary>
        /// <param name="records">A list of <see cref="Record"/>s that represent phenomenon observations.</param>
        /// <returns>A new instance of <see cref="Data"/>.</returns>
        internal DataPreprocessor Build(IList<Record> records)
        {
            return new DataPreprocessor(records, _splits, _normalization, _range);
        }
    }
}
