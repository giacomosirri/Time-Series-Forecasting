using Parquet;
using Parquet.Schema;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RNN_Model_Creation
{
    internal class Record
    {
        [ParquetColumn("Date Time")]
        public string? TimeStamp { get; set; }
        [ParquetColumn("p (mbar)")]
        public double? AirPressure { get; set; }
        [ParquetColumn("T (degC)")]
        public double? AirTemperature { get; set; }
        [ParquetColumn("Tpot (K)")]
        public double? PotentialTemperature { get; set; }
        [ParquetColumn("Tdew (degC)")]
        public double? DewPointTemperature { get; set; }
        [ParquetColumn("rh (%)")]
        public double? RelativeHumidity { get; set; }
        [ParquetColumn("VPmax (mbar)")]
        public double? SaturationWaterVaporPressure { get; set; }
        [ParquetColumn("VPact (mbar)")]
        public double? ActualWaterVaporPressure { get; set; }
        [ParquetColumn("VPdef (mbar)")]
        public double? WaterVaporPressureDeficit { get; set; }
        [ParquetColumn("sh (g/kg)")]
        public double? SpecificHumidity { get; set; }
        [ParquetColumn("H2OC (mmol/mol)")]
        public double? WaterVaporConcentration { get; set; }
        [ParquetColumn("rho (g/m**3)")]
        public double? AirDensity { get; set; }
        [ParquetColumn("wv (m/s)")]
        public double? WindVelocity { get; set; }
        [ParquetColumn("max. wv (m/s)")]
        public double? MaximumWindVelocity { get; set; }
        [ParquetColumn("wv (deg)")]
        public double? WindDirection { get; set; }
    }
}
