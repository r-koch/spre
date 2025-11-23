package dev.rkoch.spre.stock.collector.api;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandlers;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import dev.rkoch.spre.stock.collector.StockRecord;
import dev.rkoch.spre.stock.collector.exception.NoDataForDateException;
import dev.rkoch.spre.stock.collector.exception.SymbolNotFoundException;

public class NasdaqApi {

  // https://api.nasdaq.com/api/quote/tsla/historical?assetclass=stocks&fromdate=2025-08-07&limit=1&todate=2025-08-08
  // https://api.nasdaq.com/api/quote/tsla/historical?assetclass=stocks&limit=10000&fromdate=1999-11-01&todate=2025-09-07
  // https://api.nasdaq.com/api/quote/bf%25sl%25b/historical?assetclass=stocks&fromdate=2015-11-23&limit=9999&todate=2025-11-23
  private static final String API_URL = "https://api.nasdaq.com/api/quote/%s/historical?assetclass=stocks&fromdate=%s&limit=9999&todate=%s";

  private static final DateTimeFormatter MM_DD_YYYY = DateTimeFormatter.ofPattern("MM/dd/uuuu");

  private static final LocalDate DEFAULT_FROM_DATE = LocalDate.of(1999, 11, 1);

  private final Map<String, Map<LocalDate, StockRecord>> cache = new HashMap<>();

  private final HttpClient httpClient;

  private final LocalDate fromDate;

  private final LocalDate toDate = LocalDate.now();

  NasdaqApi() {
    this(DEFAULT_FROM_DATE, HttpClient.newHttpClient());
  }

  public NasdaqApi(LocalDate fromDate, HttpClient httpClient) {
    this.fromDate = fromDate;
    this.httpClient = httpClient;
  }

  private String cleanNumber(final String number) {
    return number.replace("$", "").replace(",", "");
  }

  private String getApiSymbol(final String symbol) {
    if ("fi".equalsIgnoreCase(symbol)) {
      // from 2025-11-21 fi -> fisv
      // api gets data with new symbol for old dates
      return "fisv";
    } else {
      // for symbols like bf.b the "." needs to be replaced with "%sl%" and "%" url encoded to "%25"
      return symbol.replace(".", "%25sl%25");
    }
  }

  public StockRecord getData(final LocalDate date, final String symbol) throws NoDataForDateException, SymbolNotFoundException {
    StockRecord stockRecord = getRecords(symbol).get(date);
    if (stockRecord != null) {
      return stockRecord;
    } else {
      throw new NoDataForDateException(date);
    }
  }

  private String getJsonString(final String symbol) {
    try {
      String apiSymbol = getApiSymbol(symbol);
      URI uri = getUri(apiSymbol);
      HttpRequest httpRequest = HttpRequest.newBuilder(uri).build();
      HttpResponse<String> httpResponse = httpClient.send(httpRequest, BodyHandlers.ofString());
      return httpResponse.body();
    } catch (IOException | InterruptedException e) {
      throw new RuntimeException(e);
    }
  }

  private Map<LocalDate, StockRecord> getRecords(final String symbol) throws NoDataForDateException, SymbolNotFoundException {
    Map<LocalDate, StockRecord> records = cache.get(symbol);
    if (records == null) {
      records = getRecordsFromApi(symbol);
      cache.put(symbol, records);
    }
    return records;
  }

  private Map<LocalDate, StockRecord> getRecordsFromApi(final String symbol) throws NoDataForDateException, SymbolNotFoundException {
    String source = getJsonString(symbol);
    try {
      JSONArray rows = getRows(source);
      Map<LocalDate, StockRecord> records = new HashMap<>(rows.length());
      for (int i = 0; i < rows.length(); i++) {
        JSONObject row = rows.getJSONObject(i);
        StockRecord record = rowToStockRecord(symbol, row);
        records.put(record.getLocalDate(), record);
      }
      return records;
    } catch (SymbolNotFoundException e) {
      e.setSymbol(symbol);
      throw e;
    }
  }

  private JSONArray getRows(final String source) throws NoDataForDateException, SymbolNotFoundException {
    try {
      JSONObject json = new JSONObject(source);
      JSONObject status = json.getJSONObject("status");
      if (status.getInt("rCode") == 200) {
        JSONObject data = json.getJSONObject("data");
        if (data.getInt("totalRecords") > 0) {
          return data.getJSONObject("tradesTable").getJSONArray("rows");
        } else {
          throw new NoDataForDateException(fromDate);
        }
      } else {
        if (status.getJSONArray("bCodeMessage").getJSONObject(0).getInt("code") == 1001) {
          throw new SymbolNotFoundException();
        } else {
          throw new RuntimeException(source);
        }
      }
    } catch (JSONException e) {
      throw new RuntimeException(source);
    }
  }

  private URI getUri(final String apiSymbol) {
    return URI.create(API_URL.formatted(apiSymbol, fromDate, toDate));
  }

  private StockRecord rowToStockRecord(final String symbol, final JSONObject row) {
    LocalDate rowDate = LocalDate.parse(row.getString("date"), MM_DD_YYYY);
    String close = cleanNumber(row.getString("close"));
    String high = cleanNumber(row.getString("high"));
    String low = cleanNumber(row.getString("low"));
    String open = cleanNumber(row.getString("open"));
    String volume = cleanNumber(row.getString("volume"));
    return StockRecord.of(rowDate, symbol, close, high, low, open, volume);
  }

}
