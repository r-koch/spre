package dev.rkoch.aws.s3.parquet;

import org.apache.parquet.schema.MessageType;
import blue.strategic.parquet.Dehydrator;
import blue.strategic.parquet.Hydrator;

public interface ParquetRecord<T extends ParquetRecord<T>> {

  Dehydrator<T> getDehydrator();

  Hydrator<T, T> getHydrator();

  MessageType getSchema();

}
