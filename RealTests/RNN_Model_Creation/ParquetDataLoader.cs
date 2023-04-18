using Parquet;
using Parquet.Schema;
using System.Collections;
using DataColumn = Parquet.Data.DataColumn;

namespace TimeSeriesForecasting.IO.Parquet
{
    internal class ParquetDataLoader
    {
        private readonly string _valuesFile;
        private readonly string _datesFile;

        public ParquetDataLoader(string valuesFilePath, string datesFilePath) 
        {
            _valuesFile = valuesFilePath;
            _datesFile = datesFilePath;
        }

        /// <summary>
        /// Returns all the records stored in the parquet file.
        /// </summary>
        /// <returns>A <see cref="IList"/> of <see cref="Record"/>s.</returns>
        public IList<Record> GetRecords()
        {
            IList<Record> records = new List<Record>();
            var taskNumeric = GetNumericalValues();
            taskNumeric.Wait();
            var numericalRecords = taskNumeric.Result;
            var taskDates = GetDates();
            taskDates.Wait();
            var dates = taskDates.Result;
            if (dates.Count != numericalRecords.Count) throw new Exception();
            for (int i = 0; i < numericalRecords.Count; i++)
            {
                try
                {
                    records.Add(new Record(numericalRecords[i], dates[i]));
                } 
                catch (ArgumentException)
                { 
                }
            }
            return records;
        }

        private async Task<IList<DateTime?>> GetDates()
        {
            var dates = new List<DateTime?>();
            using ParquetReader reader = await ParquetReader.CreateAsync(File.OpenRead(_datesFile));
            for (int i = 0; i < reader.RowGroupCount; i++)
            {
                using ParquetRowGroupReader rowGroupReader = reader.OpenRowGroupReader(i);
                DataField field = reader.Schema.GetDataFields()[0];
                DataColumn dc = await rowGroupReader.ReadColumnAsync(field);
                dates.AddRange((DateTime?[])dc.Data);
            }
            return dates;
        }

        private async Task<IList<IDictionary<string, double?>>> GetNumericalValues()
        {
            var records = new List<IDictionary<string, double?>>();
            using ParquetReader reader = await ParquetReader.CreateAsync(File.OpenRead(_valuesFile));
            Dictionary<string, IList<double?>> data = new();
            for (int i = 0; i < reader.RowGroupCount; i++)
            {
                using ParquetRowGroupReader rowGroupReader = reader.OpenRowGroupReader(i);
                DataField[] fields = reader.Schema.GetDataFields();
                foreach (DataField field in fields)
                {
                    DataColumn dc = await rowGroupReader.ReadColumnAsync(field);
                    data.Add(dc.Field.Name, ((double?[])dc.Data).ToList());
                }  
            }
            for (int j = 0; j < data["p (mbar)"].Count; j++)
            {
                IDictionary<string, double?> record = new Dictionary<string, double?>();
                foreach (KeyValuePair<string, IList<double?>> kvp in data)
                {
                    record.Add(kvp.Key, kvp.Value[j]);
                }
                records.Add(record);
            }
            return records;
        }
    }
}
