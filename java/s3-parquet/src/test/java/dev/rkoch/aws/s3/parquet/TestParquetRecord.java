package dev.rkoch.aws.s3.parquet;

import java.time.LocalDate;
import java.util.Objects;
import org.apache.parquet.schema.LogicalTypeAnnotation;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName;
import org.apache.parquet.schema.Types;
import blue.strategic.parquet.Dehydrator;
import blue.strategic.parquet.Hydrator;

public class TestParquetRecord implements ParquetRecord<TestParquetRecord> {

  private static final String LOCAL_DATE = "localDateValue";
  private static final String STRING = "stringValue";
  private static final String DOUBLE = "doubleValue";
  private static final String LONG = "longValue";

  private LocalDate localDateValue;
  private String stringValue;
  private double doubleValue;
  private long longValue;

  public TestParquetRecord() {

  }

  public TestParquetRecord(LocalDate localDateValue, String stringValue, double doubleValue, long longValue) {
    this.localDateValue = localDateValue;
    this.stringValue = stringValue;
    this.doubleValue = doubleValue;
    this.longValue = longValue;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (getClass() != obj.getClass()) {
      return false;
    }
    TestParquetRecord other = (TestParquetRecord) obj;
    return Double.doubleToLongBits(doubleValue) == Double.doubleToLongBits(other.doubleValue) && Objects.equals(localDateValue, other.localDateValue)
        && longValue == other.longValue && Objects.equals(stringValue, other.stringValue);
  }

  @Override
  public Dehydrator<TestParquetRecord> getDehydrator() {
    return (record, valueWriter) -> {
      valueWriter.write(LOCAL_DATE, (int) record.getLocalDateValue().toEpochDay());
      valueWriter.write(STRING, record.getStringValue());
      valueWriter.write(DOUBLE, record.getDoubleValue());
      valueWriter.write(LONG, record.getLongValue());
    };
  }

  public double getDoubleValue() {
    return doubleValue;
  }

  @Override
  public Hydrator<TestParquetRecord, TestParquetRecord> getHydrator() {
    return new Hydrator<>() {

      @Override
      public TestParquetRecord add(TestParquetRecord target, String heading, Object value) {
        switch (heading) {
          case LOCAL_DATE:
            target.setLocalDateValue(LocalDate.ofEpochDay((int) value));
            return target;
          case STRING:
            target.setStringValue((String) value);
            return target;
          case DOUBLE:
            target.setDoubleValue((double) value);
            return target;
          case LONG:
            target.setLongValue((long) value);
            return target;
          default:
            throw new IllegalArgumentException("Unexpected value: " + heading);
        }
      }

      @Override
      public TestParquetRecord finish(TestParquetRecord target) {
        return target;
      }

      @Override
      public TestParquetRecord start() {
        return new TestParquetRecord();
      }

    };
  }

  public LocalDate getLocalDateValue() {
    return localDateValue;
  }

  public long getLongValue() {
    return longValue;
  }

  @Override
  public MessageType getSchema() {
    return new MessageType("test-record", //
        Types.required(PrimitiveTypeName.INT32).as(LogicalTypeAnnotation.dateType()).named(LOCAL_DATE), //
        Types.required(PrimitiveTypeName.BINARY).as(LogicalTypeAnnotation.stringType()).named(STRING), //
        Types.required(PrimitiveTypeName.DOUBLE).named(DOUBLE), //
        Types.required(PrimitiveTypeName.INT64).named(LONG) //
    );
  }

  public String getStringValue() {
    return stringValue;
  }

  @Override
  public int hashCode() {
    return Objects.hash(doubleValue, localDateValue, longValue, stringValue);
  }

  public void setDoubleValue(double doubleValue) {
    this.doubleValue = doubleValue;
  }

  public void setLocalDateValue(LocalDate localDateValue) {
    this.localDateValue = localDateValue;
  }

  public void setLongValue(long longValue) {
    this.longValue = longValue;
  }

  public void setStringValue(String stringValue) {
    this.stringValue = stringValue;
  }

}
