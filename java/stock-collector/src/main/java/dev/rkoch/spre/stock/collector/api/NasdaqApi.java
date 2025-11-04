package dev.rkoch.spre.stock.collector.api;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandlers;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import dev.rkoch.spre.stock.collector.StockRecord;
import dev.rkoch.spre.stock.collector.exception.NoDataForDateException;
import dev.rkoch.spre.stock.collector.exception.SymbolNotExistsException;

public class NasdaqApi {

  // https://api.nasdaq.com/api/quote/tsla/historical?assetclass=stocks&fromdate=2025-08-07&limit=1&todate=2025-08-08
  // https://api.nasdaq.com/api/quote/tsla/historical?assetclass=stocks&limit=10000&fromdate=1999-11-01&todate=2025-09-07
  private static final String API_URL = "https://api.nasdaq.com/api/quote/%s/historical?assetclass=stocks&limit=10000&fromdate=%s&todate=%s";

  private static final DateTimeFormatter MM_DD_YYYY = DateTimeFormatter.ofPattern("MM/dd/uuuu");

  private static final LocalDate DEFAULT_FROM_DATE = LocalDate.of(1999, 11, 1);

  private final Map<String, List<StockRecord>> cache = new HashMap<>();

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

  public StockRecord getData(final LocalDate date, final String symbol) throws NoDataForDateException, SymbolNotExistsException {
    try {
      List<StockRecord> records = cache.get(symbol);
      if (records == null) {
        HttpRequest httpRequest = HttpRequest.newBuilder(getUri(symbol)).build();
        HttpResponse<String> httpResponse = httpClient.send(httpRequest, BodyHandlers.ofString());
        String body = httpResponse.body();
        JSONObject json = new JSONObject(body);
        try {
          JSONObject status = json.getJSONObject("status");
          if (status.getInt("rCode") == 200) {
            JSONObject data = json.getJSONObject("data");
            if (data.getInt("totalRecords") > 0) {
              JSONArray rows = data.getJSONObject("tradesTable").getJSONArray("rows");
              records = new ArrayList<>();
              for (int i = 0; i < rows.length(); i++) {
                JSONObject row = rows.getJSONObject(i);
                LocalDate rowDate = LocalDate.parse(row.getString("date"), MM_DD_YYYY);
                String close = cleanNumber(row.getString("close"));
                String high = cleanNumber(row.getString("high"));
                String low = cleanNumber(row.getString("low"));
                String open = cleanNumber(row.getString("open"));
                String volume = cleanNumber(row.getString("volume"));
                records.add(StockRecord.of(rowDate, symbol, close, high, low, open, volume));
              }
              cache.put(symbol, records);
            } else {
              throw new NoDataForDateException(fromDate);
            }
          } else {
            if (status.getJSONArray("bCodeMessage").getJSONObject(0).getInt("code") == 1001) {
              throw new SymbolNotExistsException();
            }
          }
        } catch (JSONException e) {
          throw new RuntimeException(json.toString(), e);
        }
      }
      for (StockRecord stockRecord : cache.get(symbol).reversed()) {
        if (stockRecord.getLocalDate().isEqual(date)) {
          return stockRecord;
        }
      }
      throw new NoDataForDateException(date);
    } catch (IOException | InterruptedException e) {
      throw new RuntimeException(e);
    }
  }

  private URI getUri(final String symbol) {
    String apiSymbol = symbol.replace(".", "-");
    return URI.create(API_URL.formatted(apiSymbol, fromDate, toDate));
  }

}
