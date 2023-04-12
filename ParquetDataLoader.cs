using Parquet;
using Parquet.Rows;
using Parquet.Schema;
using Parquet.Data;
using Parquet.Serialization;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.ComponentModel;
using Parquet.Thrift;
using System.Data;

namespace RNN_Model_Creation
{
    internal class ParquetDataLoader
    {
        private readonly string _filePath;

        public ParquetDataLoader(string path) 
        {
            _filePath = path;
        }

        /// <summary>
        /// Returns all the records stored in the parquet file.
        /// </summary>
        /// <returns></returns>
        public async Task GetRecords()
        {
            using Stream fs = File.OpenRead(_filePath);
            using ParquetReader reader = await ParquetReader.CreateAsync(fs);
            for (int i = 0; i < reader.RowGroupCount; i++)
            {
                using ParquetRowGroupReader rowGroupReader = reader.OpenRowGroupReader(i);
                Console.WriteLine(string.Join<DataField>(",", reader.Schema.GetDataFields()));
                foreach (DataField df in reader.Schema.GetDataFields())
                {
                    Parquet.Data.DataColumn dc = await ReadColumn(df, rowGroupReader);
                    Console.WriteLine(dc);
                }
            }
        }

        private static async Task<Parquet.Data.DataColumn> ReadColumn(DataField df, ParquetRowGroupReader groupReader)
        {
            return await groupReader.ReadColumnAsync(df);
        }
    }
}
