package dev.rkoch.aws.stock.collector;

import java.util.ArrayList;
import java.util.List;
import dev.rkoch.aws.s3.parquet.S3Parquet;

public class Symbols {

  private static final String BUCKET_NAME = "dev-rkoch-spre";

  private static final String PARQUET_KEY = "symbols/spx.parquet";

  private static final int INITIAL_CAPACITY = 503;

  private static final List<String> SYMBOLS = new ArrayList<>(INITIAL_CAPACITY);

  private static final List<SymbolRecord> RECORDS = new ArrayList<>(INITIAL_CAPACITY);

  private final S3Parquet s3Parquet;

  public Symbols(S3Parquet s3Parquet) {
    this.s3Parquet = s3Parquet;
  }

  public List<String> get() {
    if (SYMBOLS.isEmpty()) {
      SYMBOLS.addAll(getRecords().stream().map((record) -> record.getId()).toList());
    }
    return SYMBOLS;
  }

  public List<SymbolRecord> getRecords() {
    if (RECORDS.isEmpty()) {
      try {
        RECORDS.addAll(s3Parquet.read(BUCKET_NAME, PARQUET_KEY, SymbolRecord.class));
      } catch (Exception e) {
        throw new RuntimeException(e.getMessage(), e);
      }
    }
    return RECORDS;
  }

  public List<String> getAfter(final String fromSymbolAfter) {
    List<String> symbols = get();
    if (fromSymbolAfter != null && !fromSymbolAfter.isBlank()) {
      int indexOf = symbols.indexOf(fromSymbolAfter);
      if (indexOf != -1) {
        return symbols.subList(indexOf + 1, symbols.size());
      }
    }
    return symbols;
  }

  public String getLast() throws Exception {
    return get().getLast();
  }

  public void setRecords(List<SymbolRecord> records) throws Exception {
    s3Parquet.write(BUCKET_NAME, PARQUET_KEY, records);
  }

}
