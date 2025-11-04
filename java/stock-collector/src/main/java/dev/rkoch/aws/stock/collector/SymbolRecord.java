package dev.rkoch.aws.stock.collector;

import org.apache.parquet.schema.LogicalTypeAnnotation;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName;
import org.apache.parquet.schema.Types;
import blue.strategic.parquet.Dehydrator;
import blue.strategic.parquet.Hydrator;
import dev.rkoch.aws.s3.parquet.ParquetRecord;

public class SymbolRecord implements ParquetRecord<SymbolRecord> {

  private static final String ID = "id";
  private static final String SECTOR = "sector";

  private String id;
  private String sector;

  public SymbolRecord() {

  }

  public SymbolRecord(String id, String sector) {
    this.id = id;
    this.sector = sector;
  }

  @Override
  public Dehydrator<SymbolRecord> getDehydrator() {
    return (record, valueWriter) -> {
      valueWriter.write(ID, record.getId());
      valueWriter.write(SECTOR, record.getSector());
    };
  }

  @Override
  public Hydrator<SymbolRecord, SymbolRecord> getHydrator() {
    return new Hydrator<>() {

      @Override
      public SymbolRecord add(SymbolRecord target, String heading, Object value) {
        switch (heading) {
          case ID:
            target.setId((String) value);
            return target;
          case SECTOR:
            target.setSector((String) value);
            return target;
          default:
            throw new IllegalArgumentException("Unexpected value: " + heading);
        }
      }

      @Override
      public SymbolRecord finish(SymbolRecord target) {
        return target;
      }

      @Override
      public SymbolRecord start() {
        return new SymbolRecord();
      }
    };
  }

  public String getId() {
    return id;
  }

  @Override
  public MessageType getSchema() {
    return new MessageType("symbol-record", //
        Types.required(PrimitiveTypeName.BINARY).as(LogicalTypeAnnotation.stringType()).named(ID), //
        Types.required(PrimitiveTypeName.BINARY).as(LogicalTypeAnnotation.stringType()).named(SECTOR) //
    );
  }

  public String getSector() {
    return sector;
  }

  public void setId(String id) {
    this.id = id;
  }

  public void setSector(String sector) {
    this.sector = sector;
  }

}
