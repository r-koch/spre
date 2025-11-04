package dev.rkoch.aws.stock.collector;

import java.time.LocalDate;
import org.apache.parquet.schema.LogicalTypeAnnotation;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName;
import org.apache.parquet.schema.Types;
import com.crazzyghost.alphavantage.timeseries.response.StockUnit;
import blue.strategic.parquet.Dehydrator;
import blue.strategic.parquet.Hydrator;
import dev.rkoch.aws.s3.parquet.ParquetRecord;

public class StockRecord implements ParquetRecord<StockRecord> {

  private static final String LOCAL_DATE = "localDate";
  private static final String ID = "id";
  private static final String CLOSE = "close";
  private static final String HIGH = "high";
  private static final String LOW = "low";
  private static final String OPEN = "open";
  private static final String VOLUME = "volume";

  public static StockRecord of(final LocalDate localDate, final String id, final double close, final double high, final double low, final double open,
      final long volume) {
    return new StockRecord(localDate, id, close, high, low, open, volume);
  }

  public static StockRecord of(final LocalDate localDate, final String id, final String close, final String high, final String low, final String open,
      final String volume) {
    double c = Double.parseDouble(close);
    double h = Double.parseDouble(high);
    double l = Double.parseDouble(low);
    double o = Double.parseDouble(open);
    long v = parseLong(volume);
    return new StockRecord(localDate, id, c, h, l, o, v);
  }

  public static StockRecord of(final String id, final StockUnit stockUnit) {
    return new StockRecord(LocalDate.parse(stockUnit.getDate()), id, stockUnit.getClose(), stockUnit.getHigh(), stockUnit.getLow(), stockUnit.getOpen(),
        stockUnit.getVolume());
  }

  private static long parseLong(final String value) {
    try {
      return Long.parseLong(value);
    } catch (NumberFormatException e) {
      return 0L;
    }
  }

  private LocalDate localDate;
  private String id;
  private double close;
  private double high;
  private double low;
  private double open;
  private long volume;

  public StockRecord() {

  }

  public StockRecord(LocalDate localDate, String id, double close, double high, double low, double open, long volume) {
    this.localDate = localDate;
    this.id = id;
    this.close = close;
    this.high = high;
    this.low = low;
    this.open = open;
    this.volume = volume;
  }

  public double getClose() {
    return close;
  }

  @Override
  public Dehydrator<StockRecord> getDehydrator() {
    return (record, valueWriter) -> {
      valueWriter.write(LOCAL_DATE, (int) record.getLocalDate().toEpochDay());
      valueWriter.write(ID, record.getId());
      valueWriter.write(CLOSE, record.getClose());
      valueWriter.write(HIGH, record.getHigh());
      valueWriter.write(LOW, record.getLow());
      valueWriter.write(OPEN, record.getOpen());
      valueWriter.write(VOLUME, record.getVolume());
    };
  }

  public double getHigh() {
    return high;
  }

  @Override
  public Hydrator<StockRecord, StockRecord> getHydrator() {
    return new Hydrator<>() {

      @Override
      public StockRecord add(StockRecord target, String heading, Object value) {
        switch (heading) {
          case LOCAL_DATE:
            target.setLocalDate(LocalDate.ofEpochDay((int) value));
            return target;
          case ID:
            target.setId((String) value);
            return target;
          case CLOSE:
            target.setClose((double) value);
            return target;
          case HIGH:
            target.setHigh((double) value);
            return target;
          case LOW:
            target.setLow((double) value);
            return target;
          case OPEN:
            target.setOpen((double) value);
            return target;
          case VOLUME:
            target.setVolume((long) value);
            return target;
          default:
            throw new IllegalArgumentException("Unexpected value: " + heading);
        }
      }

      @Override
      public StockRecord finish(StockRecord target) {
        return target;
      }

      @Override
      public StockRecord start() {
        return new StockRecord();
      }

    };
  }

  public String getId() {
    return id;
  }

  public LocalDate getLocalDate() {
    return localDate;
  }

  public double getLow() {
    return low;
  }

  public double getOpen() {
    return open;
  }

  @Override
  public MessageType getSchema() {
    return new MessageType("stock-record", //
        Types.required(PrimitiveTypeName.INT32).as(LogicalTypeAnnotation.dateType()).named(LOCAL_DATE), //
        Types.required(PrimitiveTypeName.BINARY).as(LogicalTypeAnnotation.stringType()).named(ID), //
        Types.required(PrimitiveTypeName.DOUBLE).named(CLOSE), //
        Types.required(PrimitiveTypeName.DOUBLE).named(HIGH), //
        Types.required(PrimitiveTypeName.DOUBLE).named(LOW), //
        Types.required(PrimitiveTypeName.DOUBLE).named(OPEN), //
        Types.required(PrimitiveTypeName.INT64).named(VOLUME) //
    );
  }

  public long getVolume() {
    return volume;
  }

  public void setClose(double close) {
    this.close = close;
  }

  public void setHigh(double high) {
    this.high = high;
  }

  public void setId(String id) {
    this.id = id;
  }

  public void setLocalDate(LocalDate localDate) {
    this.localDate = localDate;
  }

  public void setLow(double low) {
    this.low = low;
  }

  public void setOpen(double open) {
    this.open = open;
  }

  public void setVolume(long volume) {
    this.volume = volume;
  }

  @Override
  public String toString() {
    return localDate + ";" + id + ";" + close + ";" + high + ";" + low + ";" + open + ";" + volume;
  }

}
