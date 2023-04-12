using Parquet.Rows;
using System;

namespace RNN_Model_Creation
{
    internal class Program
    {
        //private static readonly string FILE_PATH = "C:\\WorkingCopy\\giacomo.sirri\\test-timeseries-pred\\dataset\\" +
        //                                   "timeseries-2009-2016.snappy.parquet";
        private static readonly string FILE_PATH = "C:\\WorkingCopy\\giacomo.sirri\\old-parquet-data\\df_join.parquet";

        static void Main(string[] args)
        {
            new ParquetDataLoader(FILE_PATH).GetRecords().Wait();
        }
    }
}