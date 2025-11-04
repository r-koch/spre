package dev.rkoch.aws.news.collector;

import java.time.LocalDate;
import org.apache.parquet.schema.LogicalTypeAnnotation;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName;
import org.apache.parquet.schema.Types;
import blue.strategic.parquet.Dehydrator;
import blue.strategic.parquet.Hydrator;
import dev.rkoch.aws.s3.parquet.ParquetRecord;

public class NewsRecord implements ParquetRecord<NewsRecord> {

  private static final String LOCAL_DATE = "localDate";
  private static final String ID = "id";
  private static final String TITLE = "title";
  private static final String BODY = "body";

  public static NewsRecord of(LocalDate localDate, String id, String title, String body) {
    return new NewsRecord(localDate, id, title, body);
  }

  private LocalDate localDate;
  private String id;
  private String title;
  private String body;

  public NewsRecord() {

  }

  public NewsRecord(LocalDate localDate, String id, String title, String body) {
    this.localDate = localDate;
    this.id = id;
    this.title = title;
    this.body = body;
  }

  public String getBody() {
    return body;
  }

  @Override
  public Dehydrator<NewsRecord> getDehydrator() {
    return (record, valueWriter) -> {
      valueWriter.write(LOCAL_DATE, (int) record.getLocalDate().toEpochDay());
      valueWriter.write(ID, record.getId());
      valueWriter.write(TITLE, record.getTitle());
      valueWriter.write(BODY, record.getBody());
    };
  }

  @Override
  public Hydrator<NewsRecord, NewsRecord> getHydrator() {
    return new Hydrator<>() {

      @Override
      public NewsRecord add(NewsRecord target, String heading, Object value) {
        switch (heading) {
          case LOCAL_DATE:
            target.setLocalDate(LocalDate.ofEpochDay((int) value));
            return target;
          case ID:
            target.setId((String) value);
            return target;
          case TITLE:
            target.setTitle((String) value);
            return target;
          case BODY:
            target.setBody((String) value);
            return target;
          default:
            throw new IllegalArgumentException("Unexpected value: " + heading);
        }
      }

      @Override
      public NewsRecord finish(NewsRecord target) {
        return target;
      }

      @Override
      public NewsRecord start() {
        return new NewsRecord();
      }

    };
  }

  public String getId() {
    return id;
  }

  public LocalDate getLocalDate() {
    return localDate;
  }

  @Override
  public MessageType getSchema() {
    return new MessageType("news-record", //
        Types.required(PrimitiveTypeName.INT32).as(LogicalTypeAnnotation.dateType()).named(LOCAL_DATE), //
        Types.required(PrimitiveTypeName.BINARY).as(LogicalTypeAnnotation.stringType()).named(ID), //
        Types.required(PrimitiveTypeName.BINARY).as(LogicalTypeAnnotation.stringType()).named(TITLE), //
        Types.required(PrimitiveTypeName.BINARY).as(LogicalTypeAnnotation.stringType()).named(BODY) //
    );
  }

  public String getTitle() {
    return title;
  }

  public void setBody(String body) {
    this.body = body;
  }

  public void setId(String id) {
    this.id = id;
  }

  public void setLocalDate(LocalDate localDate) {
    this.localDate = localDate;
  }

  public void setTitle(String title) {
    this.title = title;
  }

}
