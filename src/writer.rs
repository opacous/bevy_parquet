use std::fs::File;
use std::sync::Arc;
use arrow::datatypes::SchemaRef;
use arrow::array::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::format::FileMetaData;
use parquet::file::properties::WriterProperties;
use crate::ParquetError;

pub struct ParquetWriter {
    writer: ArrowWriter<File>,
}

impl ParquetWriter {
    pub fn new(
        path: String,
        schema: SchemaRef,
        properties: WriterProperties,
    ) -> Result<Self, ParquetError> {
        let file = File::create(path)
            .map_err(ParquetError::Io)?;
            
        let writer = ArrowWriter::try_new(file, schema, Some(properties))
            .map_err(|e| ParquetError::ParquetWrite(e.to_string()))?;

        Ok(Self { writer })
    }

    pub fn write_batch(&mut self, batch: &RecordBatch) -> Result<(), ParquetError> {
        self.writer
            .write(batch)
            .map_err(|e| ParquetError::ParquetWrite(e.to_string()))
    }

    pub fn finalize(self) -> Result<FileMetaData, ParquetError> {
        self.writer
            .close()
            .map_err(|e| ParquetError::ParquetWrite(e.to_string()))
    }
}
