package dev.rkoch.aws.s3.parquet;

import java.time.LocalDate;
import java.util.List;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class TestS3Parquet {

  @Test
  public void testReadWrite() throws Exception {
    List<TestParquetRecord> writeRecords =
        List.of(new TestParquetRecord(LocalDate.parse("2011-11-11"), "test", 1.1, 11), new TestParquetRecord(LocalDate.parse("2012-12-12"), "test", 1.2, 12));

    S3Parquet s3Parquet = new S3Parquet();

    s3Parquet.write("dev-rkoch-spre-test", "test.parquet", writeRecords);

    List<TestParquetRecord> readRecords = s3Parquet.read("dev-rkoch-spre-test", "test.parquet", TestParquetRecord.class);

    Assertions.assertIterableEquals(writeRecords, readRecords);
  }

}
