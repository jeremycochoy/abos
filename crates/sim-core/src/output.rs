use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Int64Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use crate::exchange::{L1Snapshot, TradeRecord};

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
