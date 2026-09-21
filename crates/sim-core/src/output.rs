use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    Float64Array, Int64Array, StringArray, TimestampMillisecondArray, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use crate::exchange::{L1Bucket, L1Snapshot, TradeRecord};

/// Write trade records to a Parquet file.
///
/// # Errors
/// Returns an error if the file cannot be created or written.
pub fn write_trades(
    path: &Path,
    trades: &[TradeRecord],
    tick_size: i64,
    lot_size: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("timestamp", DataType::UInt64, false),
            Field::new("symbol", DataType::UInt64, false),
            Field::new("price", DataType::Int64, false),
            Field::new("qty", DataType::UInt64, false),
            Field::new("aggressor_side", DataType::Utf8, false),
            Field::new("maker_order_id", DataType::UInt64, false),
            Field::new("taker_order_id", DataType::UInt64, false),
        ],
        HashMap::from([
            ("tick_size".into(), tick_size.to_string()),
            ("lot_size".into(), lot_size.to_string()),
        ]),
    ));

    let batch = RecordBatch::try_new(schema.clone(), vec![
        Arc::new(UInt64Array::from_iter_values(trades.iter().map(|t| t.timestamp))),
        Arc::new(UInt64Array::from_iter_values(trades.iter().map(|t| u64::from(t.symbol)))),
        Arc::new(Int64Array::from_iter_values(trades.iter().map(|t| t.price))),
        Arc::new(UInt64Array::from_iter_values(trades.iter().map(|t| t.qty))),
        Arc::new(StringArray::from_iter_values(trades.iter().map(|t| match t.aggressor_side {
            cda_engine::Side::Bid => "bid",
            cda_engine::Side::Ask => "ask",
        }))),
        Arc::new(UInt64Array::from_iter_values(trades.iter().map(|t| t.maker_order_id))),
        Arc::new(UInt64Array::from_iter_values(trades.iter().map(|t| t.taker_order_id))),
    ])?;

    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Write L1 snapshots to a Parquet file.
///
/// # Errors
/// Returns an error if the file cannot be created or written.
pub fn write_l1_snapshots(
    path: &Path,
    snapshots: &[L1Snapshot],
    tick_size: i64,
    lot_size: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("timestamp", DataType::UInt64, false),
            Field::new("symbol", DataType::UInt64, false),
            Field::new("bid_price", DataType::Int64, false),
            Field::new("ask_price", DataType::Int64, false),
            Field::new("bid_volume", DataType::UInt64, false),
            Field::new("ask_volume", DataType::UInt64, false),
            Field::new("last_trade_price", DataType::Int64, false),
        ],
        HashMap::from([
            ("tick_size".into(), tick_size.to_string()),
            ("lot_size".into(), lot_size.to_string()),
        ]),
    ));

    let batch = RecordBatch::try_new(schema.clone(), vec![
        Arc::new(UInt64Array::from_iter_values(snapshots.iter().map(|s| s.timestamp))),
        Arc::new(UInt64Array::from_iter_values(snapshots.iter().map(|s| u64::from(s.symbol)))),
        Arc::new(Int64Array::from_iter_values(snapshots.iter().map(|s| s.bid_price))),
        Arc::new(Int64Array::from_iter_values(snapshots.iter().map(|s| s.ask_price))),
        Arc::new(UInt64Array::from_iter_values(snapshots.iter().map(|s| s.bid_volume))),
        Arc::new(UInt64Array::from_iter_values(snapshots.iter().map(|s| s.ask_volume))),
        Arc::new(Int64Array::from_iter_values(snapshots.iter().map(|s| s.last_trade_price))),
    ])?;

    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Write L1 bucket aggregates to a Parquet file.
///
/// The schema carries `tick_size`, `lot_size` and `bucket_ns` as metadata.
/// `quote_volume` lands as `Float64`: a value past 2^53 keeps its scale and
/// loses its last bits, like the float pipeline that reads it. Writes in
/// batches of 1 million rows to limit peak arrow memory usage.
///
/// # Errors
/// Returns an error if the file cannot be created or written.
pub fn write_l1_buckets(
    path: &Path,
    buckets: &[L1Bucket],
    tick_size: i64,
    lot_size: u64,
    bucket_ns: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    const CHUNK: usize = 1_000_000;

    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("bucket_start", DataType::UInt64, false),
            Field::new("symbol", DataType::UInt64, false),
            Field::new("first_bid", DataType::Int64, false),
            Field::new("min_bid", DataType::Int64, false),
            Field::new("last_bid", DataType::Int64, false),
            Field::new("first_ask", DataType::Int64, false),
            Field::new("max_ask", DataType::Int64, false),
            Field::new("last_ask", DataType::Int64, false),
            Field::new("volume", DataType::UInt64, false),
            Field::new("quote_volume", DataType::Float64, false),
        ],
        HashMap::from([
            ("tick_size".into(), tick_size.to_string()),
            ("lot_size".into(), lot_size.to_string()),
            ("bucket_ns".into(), bucket_ns.to_string()),
        ]),
    ));

    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

    for chunk in buckets.chunks(CHUNK) {
        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(UInt64Array::from_iter_values(chunk.iter().map(|b| b.bucket_start))),
            Arc::new(UInt64Array::from_iter_values(chunk.iter().map(|b| u64::from(b.symbol)))),
            Arc::new(Int64Array::from_iter_values(chunk.iter().map(|b| b.first_bid))),
            Arc::new(Int64Array::from_iter_values(chunk.iter().map(|b| b.min_bid))),
            Arc::new(Int64Array::from_iter_values(chunk.iter().map(|b| b.last_bid))),
            Arc::new(Int64Array::from_iter_values(chunk.iter().map(|b| b.first_ask))),
            Arc::new(Int64Array::from_iter_values(chunk.iter().map(|b| b.max_ask))),
            Arc::new(Int64Array::from_iter_values(chunk.iter().map(|b| b.last_ask))),
            Arc::new(UInt64Array::from_iter_values(chunk.iter().map(|b| b.volume))),
            #[allow(clippy::cast_precision_loss)]
            Arc::new(Float64Array::from_iter_values(
                chunk.iter().map(|b| b.quote_volume as f64),
            )),
        ])?;
        writer.write(&batch)?;
    }

    writer.close()?;
    Ok(())
}

/// A single 1-minute OHLCV candle with nanosecond timestamp.
#[derive(Debug, Clone, Copy)]
pub struct Candle {
    /// Nanoseconds since Unix epoch (can be negative for pre-1970 dates).
    pub ts_nanos: i64,
    /// Opening price.
    pub open: f64,
    /// High price.
    pub high: f64,
    /// Low price.
    pub low: f64,
    /// Closing price.
    pub close: f64,
    /// Total trade volume in the period.
    pub volume: f64,
    /// Total quote volume (notional) in the period.
    pub quote_volume: f64,
}

/// Write 1-minute kline candles to a Parquet file.
///
/// Matches Binance kline parquet format: data columns first (`open`, `high`,
/// `low`, `close`, `volume`, `quote_volume` as `Float64`), then `ts`
/// (`Timestamp(Millisecond, None)`) as the pandas-style index column.
///
/// Writes in batches of 1 million rows to limit peak arrow memory usage.
///
/// # Errors
/// Returns an error if the file cannot be created or written.
pub fn write_klines(
    path: &Path,
    candles: &[Candle],
) -> Result<(), Box<dyn std::error::Error>> {
    const CHUNK: usize = 1_000_000;

    // Column order matches Binance: data columns first, ts last (as index).
    let pandas_metadata = r#"{"index_columns":["ts"],"column_indexes":[{"name":null,"field_name":null,"pandas_type":"unicode","numpy_type":"object","metadata":{"encoding":"UTF-8"}}],"columns":[{"name":"open","field_name":"open","pandas_type":"float64","numpy_type":"float64","metadata":null},{"name":"high","field_name":"high","pandas_type":"float64","numpy_type":"float64","metadata":null},{"name":"low","field_name":"low","pandas_type":"float64","numpy_type":"float64","metadata":null},{"name":"close","field_name":"close","pandas_type":"float64","numpy_type":"float64","metadata":null},{"name":"volume","field_name":"volume","pandas_type":"float64","numpy_type":"float64","metadata":null},{"name":"quote_volume","field_name":"quote_volume","pandas_type":"float64","numpy_type":"float64","metadata":null},{"name":"ts","field_name":"ts","pandas_type":"datetime","numpy_type":"datetime64[ms]","metadata":null}],"creator":{"library":"sim-core","version":"0.1.0"}}"#;

    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("open", DataType::Float64, false),
            Field::new("high", DataType::Float64, false),
            Field::new("low", DataType::Float64, false),
            Field::new("close", DataType::Float64, false),
            Field::new("volume", DataType::Float64, false),
            Field::new("quote_volume", DataType::Float64, false),
            Field::new("ts", DataType::Timestamp(TimeUnit::Millisecond, None), false),
        ],
        HashMap::from([("pandas".into(), pandas_metadata.into())]),
    ));

    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

    for chunk in candles.chunks(CHUNK) {
        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(Float64Array::from_iter_values(chunk.iter().map(|c| c.open))),
            Arc::new(Float64Array::from_iter_values(chunk.iter().map(|c| c.high))),
            Arc::new(Float64Array::from_iter_values(chunk.iter().map(|c| c.low))),
            Arc::new(Float64Array::from_iter_values(chunk.iter().map(|c| c.close))),
            Arc::new(Float64Array::from_iter_values(chunk.iter().map(|c| c.volume))),
            Arc::new(Float64Array::from_iter_values(chunk.iter().map(|c| c.quote_volume))),
            Arc::new(TimestampMillisecondArray::from_iter_values(
                chunk.iter().map(|c| c.ts_nanos / 1_000_000),
            )),
        ])?;
        writer.write(&batch)?;
    }

    writer.close()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    #[test]
    fn l1_buckets_round_trip_through_parquet() {
        let mut first = L1Bucket::new(0, 1);
        first.absorb(101, 103, 4, 6);
        let mut second = L1Bucket::new(60_000_000_000, 1);
        second.absorb(0, 0, 0, 0);
        second.absorb(99, 105, 1, 2);
        let path = std::env::temp_dir()
            .join(format!("l1_buckets_{}.parquet", std::process::id()));
        write_l1_buckets(&path, &[first, second], 100, 5, 60_000_000_000).unwrap();

        let file = File::open(&path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let metadata = reader.schema().metadata().clone();
        assert_eq!(metadata["tick_size"], "100");
        assert_eq!(metadata["lot_size"], "5");
        assert_eq!(metadata["bucket_ns"], "60000000000");

        let batches: Vec<RecordBatch> =
            reader.build().unwrap().collect::<Result<_, _>>().unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 2);
        let int_column = |name: &str| -> Vec<i64> {
            let column = batch.column_by_name(name).unwrap();
            let values = column.as_any().downcast_ref::<Int64Array>().unwrap();
            (0..values.len()).map(|i| values.value(i)).collect()
        };
        assert_eq!(int_column("min_bid"), vec![101, 99]);
        assert_eq!(int_column("max_ask"), vec![103, 105]);
        assert_eq!(int_column("last_bid"), vec![101, 99]);
        let volume = batch.column_by_name("volume").unwrap();
        let volume = volume.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!((volume.value(0), volume.value(1)), (10, 3));
        let notional = batch.column_by_name("quote_volume").unwrap();
        let notional = notional.as_any().downcast_ref::<Float64Array>().unwrap();
        assert!((notional.value(0) - 1022.0).abs() < 1e-9);
        assert!((notional.value(1) - 309.0).abs() < 1e-9);
    }
}
