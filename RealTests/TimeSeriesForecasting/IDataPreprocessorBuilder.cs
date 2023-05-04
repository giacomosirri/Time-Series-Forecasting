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
        /// Normalize the data.
        /// </summary>
        /// <param name="normalization">An existing <see cref="NormalizationMethod"/>.</param>
        /// <returns></returns>
        internal IDataPreprocessorBuilder Normalize(NormalizationMethod normalization);

        internal IDataPreprocessorBuilder Split((int training, int validation, int test) splits);

        internal IDataPreprocessorBuilder AddDateRange((DateTime? firstDate, DateTime? lastDate) range);

        internal DataPreprocessor Build(IList<Record> records);
    }
}
