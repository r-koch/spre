package dev.rkoch.aws.historic.stock.collector;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.naming.LimitExceededException;
import dev.rkoch.aws.collector.utils.State;
import dev.rkoch.aws.s3.parquet.S3Parquet;
import dev.rkoch.aws.stock.collector.StockRecord;
import dev.rkoch.aws.stock.collector.Symbols;
import dev.rkoch.aws.stock.collector.api.AlphaVantageApi;
import software.amazon.awssdk.http.urlconnection.UrlConnectionHttpClient;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;

public class HistoricStockCollector {

  private static final String BUCKET_NAME = "dev-rkoch-spre";

  private static final String PARQUET_KEY = "raw/stock/localDate=%s/data.parquet";

  public static void main(String[] args) {
    HistoricStockCollector collector = new HistoricStockCollector();
    // collector.collect();
    collector.store();
  }

  private S3Client s3Client;

  private S3Client getS3Client() {
    if (s3Client == null) {
      s3Client = S3Client.builder().region(Region.EU_WEST_1).httpClientBuilder(UrlConnectionHttpClient.builder()).build();
    }
    return s3Client;
  }

  private void store() {
    List<StockRecord> stockRecords = read(Path.of(System.getProperty("user.dir"), "data.txt"));
    Map<LocalDate, Map<String, StockRecord>> map = toMap(stockRecords);
    try (State state = new State(getS3Client(), BUCKET_NAME)) {
      LocalDate date = state.getAvStartDate();
      LocalDate endDate = state.getNasdaqStartDate();
      for (; date.isBefore(endDate); date = date.plusDays(1)) {
        Map<String, StockRecord> map2 = map.get(date);
        if (map2 != null) {
          List<StockRecord> records = new ArrayList<>();
          Symbols symbols = new Symbols(getS3Parquet());
          for (String symbol : symbols.get()) {
            StockRecord stockRecord = map2.get(symbol);
            if (stockRecord == null) {
              stockRecord = StockRecord.of(date, symbol, 0, 0, 0, 0, 0);
            }
            records.add(stockRecord);
          }
          insert(date, records);
        }
      }
    }
  }

  private Map<LocalDate, Map<String, StockRecord>> toMap(List<StockRecord> stockRecords) {
    Map<LocalDate, Map<String, StockRecord>> map = new HashMap<>();
    for (StockRecord stockRecord : stockRecords) {
      Map<String, StockRecord> dateMap = map.get(stockRecord.getLocalDate());
      if (dateMap == null) {
        dateMap = new HashMap<>();
      }
      dateMap.put(stockRecord.getId(), stockRecord);
      map.put(stockRecord.getLocalDate(), dateMap);
    }
    return map;
  }

  private List<StockRecord> read(Path path) {
    try {
      List<String> lines = Files.readAllLines(path);
      List<StockRecord> records = new ArrayList<>(lines.size());
      for (String line : lines) {
        String[] split = line.split(";");
        records.add(StockRecord.of(LocalDate.parse(split[0]), split[1], split[2], split[3], split[4], split[5], split[6]));
      }
      return records;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private AlphaVantageApi alphaVantageApi;

  public HistoricStockCollector() {

  }

  public void collect() {
    try (State state = new State(getS3Client(), BUCKET_NAME)) {
      Symbols symbols = new Symbols(getS3Parquet());
      String lastSymbol = state.getLastProcessedStock();
      for (String symbol : symbols.getAfter(lastSymbol)) {
        try {
          List<StockRecord> records = getAlphaVantageApi().getData(symbol);
          write(records);
          System.out.println("written %s".formatted(symbol));
          state.setLastProcessedStock(symbol);
        } catch (LimitExceededException e) {
          System.out.println("limit exceeded");
          return;
        }
      }
    }
  }

  private void write(List<StockRecord> records) {
    try {
      Path file = Path.of(System.getProperty("user.dir"), "data.txt");
      try (BufferedWriter writer = Files.newBufferedWriter(file, StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
        for (StockRecord record : records) {
          writer.write(record.toString());
          writer.newLine();
        }
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  private AlphaVantageApi getAlphaVantageApi() {
    if (alphaVantageApi == null) {
      alphaVantageApi = new AlphaVantageApi();
    }
    return alphaVantageApi;
  }

  private S3Parquet s3Parquet;

  private S3Parquet getS3Parquet() {
    if (s3Parquet == null) {
      s3Parquet = new S3Parquet(getS3Client());
    }
    return s3Parquet;
  }

  private void insert(final LocalDate date, final List<StockRecord> records) {
    try {
      getS3Parquet().write(BUCKET_NAME, PARQUET_KEY.formatted(date), records);
      System.out.println("%s inserted".formatted(date));
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

}
