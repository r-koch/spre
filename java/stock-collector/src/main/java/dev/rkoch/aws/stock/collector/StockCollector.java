package dev.rkoch.aws.stock.collector;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import javax.naming.LimitExceededException;
import com.amazonaws.services.lambda.runtime.LambdaLogger;
import com.amazonaws.services.lambda.runtime.logging.LogLevel;
import dev.rkoch.aws.collector.utils.State;
import dev.rkoch.aws.stock.collector.api.AlphaVantageApi;
import dev.rkoch.aws.stock.collector.api.NasdaqApi;
import dev.rkoch.aws.stock.collector.exception.NoDataForDateException;
import dev.rkoch.aws.stock.collector.exception.SymbolNotExistsException;

public class StockCollector {

  private static final String BUCKET_NAME = "dev-rkoch-spre";

  private static final String PARQUET_KEY = "raw/stock/localDate=%s/data.parquet";

  private final LambdaLogger logger;

  private final Handler handler;

  private AlphaVantageApi alphaVantageApi;

  private NasdaqApi nasdaqApi;

  public StockCollector(LambdaLogger logger, Handler handler) {
    this.logger = logger;
    this.handler = handler;
  }

  public void collect() {
    try {
      collect(new Symbols(handler.getS3Parquet()).get());
    } catch (Exception e) {
      logger.log(e.getMessage(), LogLevel.ERROR);
    }
  }

  private void collect(final List<String> symbols) {
    try (State state = new State(handler.getS3Client(), BUCKET_NAME)) {
      LocalDate limitExceeded = state.getAvLimitExceededDate();
      LocalDate now = LocalDate.now();
      if (limitExceeded == null || now.isAfter(limitExceeded)) {
        LocalDate date = getStartDate(state);
        for (; date.isBefore(now); date = date.plusDays(1)) {
          try {
            List<StockRecord> records = getData(date, symbols);
            insert(date, records);
            state.setLastAddedStockDate(date);
            logger.log("%s inserted".formatted(date), LogLevel.INFO);
          } catch (NoDataForDateException e) {
            continue;
          } catch (LimitExceededException e) {
            logger.log(e.getMessage(), LogLevel.ERROR);
            state.setAvLimitExceededDate(now);
            return;
          } catch (Exception e) {
            logger.log(e.getMessage(), LogLevel.ERROR);
            return;
          }
        }
      }
    }
  }

  private AlphaVantageApi getAlphaVantageApi() {
    if (alphaVantageApi == null) {
      alphaVantageApi = new AlphaVantageApi();
    }
    return alphaVantageApi;
  }

  private List<StockRecord> getData(final LocalDate date, final List<String> symbols) throws LimitExceededException, NoDataForDateException {
    List<StockRecord> records = new ArrayList<>();
    boolean isTradingDay = false;
    for (String symbol : symbols) {
      try {
        try {
          records.add(getNasdaqApi(date).getData(date, symbol));
          isTradingDay = true;
          try {
            Thread.sleep(11L);
          } catch (InterruptedException e) {
            logger.log(e.getMessage(), LogLevel.ERROR);
          }
        } catch (SymbolNotExistsException e) {
          records.add(getAlphaVantageApi().getData(date, symbol));
          isTradingDay = true;
        }
      } catch (NoDataForDateException e) {
        records.add(StockRecord.of(date, symbol, 0, 0, 0, 0, 0));
      }
      if (!isTradingDay) {
        throw new NoDataForDateException();
      }
      logger.log("%s collected %s".formatted(date, symbol), LogLevel.TRACE);
    }
    return records;
  }

  private NasdaqApi getNasdaqApi(final LocalDate date) {
    if (nasdaqApi == null) {
      nasdaqApi = new NasdaqApi(date, handler.getHttpClient());
    }
    return nasdaqApi;
  }

  private LocalDate getStartDate(final State state) {
    LocalDate lastAddedStockDate = state.getLastAddedStockDate();
    if (lastAddedStockDate == null) {
      LocalDate startDate = LocalDate.now().minusYears(10).minusDays(1);
      state.setNasdaqStartDate(startDate);
      return startDate;
    } else {
      return lastAddedStockDate.plusDays(1);
    }
  }

  private void insert(final LocalDate date, final List<StockRecord> records) throws Exception {
    handler.getS3Parquet().write(BUCKET_NAME, PARQUET_KEY.formatted(date), records);
  }

}
