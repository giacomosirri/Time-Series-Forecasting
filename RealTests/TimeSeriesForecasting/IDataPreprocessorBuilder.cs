using TimeSeriesForecasting.IO;
using static TimeSeriesForecasting.DataPreprocessor;

namespace TimeSeriesForecasting
{
    /// <summary>
    /// Creates a contract which every <see cref="DataPreprocessor"/> builder must adhere to.
    /// Data preprocessing is the set of operations performed on raw data to make it suitable
    /// for machine learning and deep learning models. There are a lot of different operations
    /// that can be grouped under this term. This builder allows to perform some common data
    /// preprocessing operations before the <see cref="DataPreprocessor"/> instance is even 
    /// created.
    /// </summary>
    internal interface IDataPreprocessorBuilder
    {
        /// <summary>
        /// Splits the data into training, validation and test sets.
        /// </summary>
        /// <param name="splits">A <see cref="Tuple"/> with the percentages of values to be included in the 
        /// training, validation and test set respectively.</param>
        /// <returns>This instance of <see cref="DataPreprocessorBuilder"/>.</returns>
        internal IDataPreprocessorBuilder Split((int training, int validation, int test) splits);

        /// <summary>
        /// Normalize the data.
        /// </summary>
        /// <param name="normalization">The normalization method.</param>
        /// <returns>This instance of <see cref="DataPreprocessorBuilder"/>.</returns>
        internal IDataPreprocessorBuilder Normalize(NormalizationMethod normalization);

        /// <summary>
        /// Clips the data to the given date boundaries.
        /// </summary>
        /// <param name="range">A <see cref="Tuple"/> that contains the first and last date to be included in the
        /// processed data. It might be useful to speed up processing if the dataset contains dozens of thousands of 
        /// observations or even more.</param>
        /// <returns>This instance of <see cref="DataPreprocessorBuilder"/>.</returns>
        internal IDataPreprocessorBuilder AddDateRange((DateTime? firstDate, DateTime? lastDate) range);

        /// <summary>
        /// Creates a new instance of DataPreprocessor, with the custom parameters previously specified.
        /// </summary>
        /// <param name="records">A list of <see cref="Record"/>s that represent phenomenon observations.</param>
        /// <returns>A new instance of <see cref="Data"/>.</returns>
        internal DataPreprocessor Build(IList<Record> records);
    }
}
