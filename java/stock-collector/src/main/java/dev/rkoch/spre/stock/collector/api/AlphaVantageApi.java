package dev.rkoch.spre.stock.collector.api;

import java.time.LocalDate;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.naming.LimitExceededException;
import com.crazzyghost.alphavantage.AlphaVantage;
import com.crazzyghost.alphavantage.Config;
import com.crazzyghost.alphavantage.parameters.OutputSize;
import com.crazzyghost.alphavantage.timeseries.response.StockUnit;
import com.crazzyghost.alphavantage.timeseries.response.TimeSeriesResponse;
import dev.rkoch.spre.stock.collector.StockRecord;
import dev.rkoch.spre.stock.collector.exception.NoDataForDateException;

public class AlphaVantageApi {

  private static final String ALPHAVANTAGE_API_KEY = "ALPHAVANTAGE_API_KEY";

  private final AlphaVantage alphaVantage;

  private final Map<String, List<StockUnit>> cache = new HashMap<>();

  public AlphaVantageApi() {
    alphaVantage = AlphaVantage.api();
    alphaVantage.init(Config.builder().key(System.getenv(ALPHAVANTAGE_API_KEY)).build());
  }

  public StockRecord getData(final LocalDate date, final String symbol) throws LimitExceededException, NoDataForDateException {
    for (StockUnit stockUnit : getStockUnits(symbol)) {
      if (LocalDate.parse(stockUnit.getDate()).isEqual(date)) {
        return StockRecord.of(symbol, stockUnit);
      }
    }
    throw new NoDataForDateException("no data found for %s on %s".formatted(symbol, date));
  }

  public List<StockRecord> getData(final String symbol) throws LimitExceededException {
    return getStockUnits(symbol).stream().map((stockUnit) -> StockRecord.of(symbol, stockUnit)).toList();
  }

  private List<StockUnit> getStockUnits(final String symbol) throws LimitExceededException {
    List<StockUnit> stockUnits = cache.get(symbol);
    if (stockUnits == null) {
      String apiSymbol = symbol.replace(".", "-");
      TimeSeriesResponse response = alphaVantage.timeSeries().daily().forSymbol(apiSymbol).outputSize(OutputSize.FULL).fetchSync();
      String errorMessage = response.getErrorMessage();
      if (errorMessage != null && !errorMessage.isBlank()) {
        throw new LimitExceededException("alphavantage limit exceeded for %s".formatted(LocalDate.now()));
      }
      stockUnits = response.getStockUnits();
      cache.put(symbol, stockUnits);
    }
    return stockUnits;
  }

}
