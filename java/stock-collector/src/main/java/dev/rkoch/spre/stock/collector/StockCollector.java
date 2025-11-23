package dev.rkoch.spre.stock.collector;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import com.amazonaws.services.lambda.runtime.Context;
import com.amazonaws.services.lambda.runtime.LambdaLogger;
import com.amazonaws.services.lambda.runtime.logging.LogLevel;
import dev.rkoch.spre.collector.utils.Environment;
import dev.rkoch.spre.collector.utils.State;
import dev.rkoch.spre.stock.collector.api.NasdaqApi;
import dev.rkoch.spre.stock.collector.exception.NoDataForDateException;
import dev.rkoch.spre.stock.collector.exception.SymbolNotExistsException;

public class StockCollector {

  private static final String BUCKET_NAME = Environment.get("BUCKET_NAME", "dev-rkoch-spre");

  private static final String LAST_ADDED = "lastAdded";

  private static final long MIN_REMAINING_TIME_MILLIS = Environment.get("MIN_REMAINING_TIME_MILLIS", 30_000); // 30 sec

  private static final String PARQUET_KEY = Environment.get("PARQUET_KEY", "raw/stock/localDate=%s/data.parquet");

  private static final String STATE_KEY = Environment.get("STATE_KEY", "metadata/stock_collector_state.json");

  private final Context context;

  private final Handler handler;

  private final LambdaLogger logger;

  private NasdaqApi nasdaqApi;

  private List<String> symbols;

  public StockCollector(Context context, Handler handler) {
    this.context = context;
    this.logger = context.getLogger();
    this.handler = handler;
  }

  public void collect() {
    try (State state = new State(handler.getS3Client(), BUCKET_NAME, STATE_KEY)) {
      LocalDate start = getStartDate(state);
      LocalDate end = LocalDate.now();
      collect(state, start, end);
    }
  }

  private void collect(final State state, final LocalDate start, final LocalDate end) {
    for (LocalDate date = start; continueExecution() && date.isBefore(end); date = date.plusDays(1)) {
      try {
        List<StockRecord> records = getData(date);
        if (records.isEmpty()) {
          logger.log("%s no data".formatted(date), LogLevel.INFO);
        } else {
          insert(date, records);
          logger.log("%s inserted".formatted(date), LogLevel.INFO);
        }
        state.setDate(LAST_ADDED, date);
      } catch (NoDataForDateException e) {
        continue;
      } catch (Throwable t) {
        logger.log(t.getMessage(), LogLevel.ERROR);
        return;
      }
    }
  }

  private boolean continueExecution() {
    return context.getRemainingTimeInMillis() >= MIN_REMAINING_TIME_MILLIS;
  }

  private List<StockRecord> getData(final LocalDate date) throws NoDataForDateException, SymbolNotExistsException {
    List<StockRecord> records = new ArrayList<>();
    boolean dataFoundForDate = false;
    for (String symbol : getSymbols()) {
      try {
        records.add(getNasdaqApi(date).getData(date, symbol));
        dataFoundForDate = true;
        try {
          Thread.sleep(11L);
        } catch (InterruptedException e) {
          logger.log(e.getMessage(), LogLevel.ERROR);
        }
      } catch (NoDataForDateException e) {
        records.add(StockRecord.of(date, symbol, 0, 0, 0, 0, 0));
      }
      if (!dataFoundForDate) {
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
    LocalDate lastAddedStockDate = state.getDate(LAST_ADDED);
    if (lastAddedStockDate == null) {
      throw new IllegalStateException("%s not found in %s".formatted(LAST_ADDED, STATE_KEY));
    } else {
      return lastAddedStockDate.plusDays(1);
    }
  }

  private List<String> getSymbols() {
    if (symbols == null) {
      symbols = new Symbols(handler.getS3Parquet()).get();
    }
    return symbols;
  }

  private void insert(final LocalDate date, final List<StockRecord> records) throws Exception {
    handler.getS3Parquet().write(BUCKET_NAME, PARQUET_KEY.formatted(date), records);
  }

}
