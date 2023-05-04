using System.Reflection.Metadata.Ecma335;
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

        IDataPreprocessorBuilder IDataPreprocessorBuilder.AddDateRange((DateTime? firstDate, DateTime? lastDate) range)
        {
            _range = range;
            return this;
        }

        IDataPreprocessorBuilder IDataPreprocessorBuilder.Normalize(DataPreprocessor.NormalizationMethod normalization)
        {
            _normalization = normalization;
            return this;
        }

        IDataPreprocessorBuilder IDataPreprocessorBuilder.Split((int training, int validation, int test) splits)
        {
            _splits = splits;
            return this;
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
        /// processed data. It might be useful to speed up processing if the dataset contains dozens of thousands of 
        /// observations or even more.</param>
        DataPreprocessor IDataPreprocessorBuilder.Build(IList<Record> records)
        {
            return new DataPreprocessor(records, _splits, _normalization, _range);
        }
    }
}
