package dev.rkoch.spre.news.collector;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandlers;
import java.time.Duration;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import javax.naming.LimitExceededException;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import com.amazonaws.services.lambda.runtime.Context;
import com.amazonaws.services.lambda.runtime.LambdaLogger;
import com.amazonaws.services.lambda.runtime.logging.LogLevel;
import dev.rkoch.spre.collector.utils.Environment;
import dev.rkoch.spre.collector.utils.State;

public class NewsCollector {

  // https://open-platform.theguardian.com/documentation/search
  // https://content.guardianapis.com/search?api-key=<API_KEY>&show-fields=bodyText&lang=en&from-date=2005-08-15&to-date=2005-08-15&page=2
  private static final String API_URL =
      "https://content.guardianapis.com/search?api-key=%s&show-fields=bodyText&lang=en&page-size=50&from-date=%s&to-date=%s&page=%s";

  private static final long API_CALL_MIN_FREQUENCY_MILLIS = Environment.get("API_CALL_MIN_FREQUENCY_MILLIS", 1010);

  private static final LocalDate AV_START_DATE = LocalDate.parse(Environment.get("AV_START_DATE", "1999-11-01"));

  private static final String BUCKET_NAME = Environment.get("BUCKET_NAME", "dev-rkoch-spre");

  private static final String LAST_ADDED = "lastAdded";

  private static final long MIN_REMAINING_TIME_MILLIS = Environment.get("MIN_REMAINING_TIME_MILLIS", 30_000); // 30 sec

  private static final String PARQUET_KEY = Environment.get("STATE_KEY", "raw/news/localDate=%s/data.parquet");

  private static final String PILLAR_ID = "pillar/news";

  private static final String STATE_KEY = Environment.get("STATE_KEY", "metadata/news_collector_state.json");

  private final Context context;

  private final Handler handler;

  private final LambdaLogger logger;

  private long lastApiCallMillis = 0;

  public NewsCollector(Context context, Handler handler) {
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

  public void collect(final State state, final LocalDate start, final LocalDate end) {
    for (LocalDate date = start; continueExecution() && date.isBefore(end); date = date.plusDays(1)) {
      try {
        List<NewsRecord> records = getData(date);
        if (records.isEmpty()) {
          logger.log("%s no data".formatted(date), LogLevel.INFO);
        } else {
          insert(date, records);
          logger.log("%s inserted".formatted(date), LogLevel.INFO);
        }
        state.setDate(LAST_ADDED, date);
      } catch (LimitExceededException e) {
        logger.log("%s theguardian limit exceeded? %s".formatted(date, e.getMessage()), LogLevel.INFO);
        return;
      } catch (Throwable t) {
        logger.log(t.getMessage(), LogLevel.ERROR);
        return;
      }
    }
  }

  private boolean continueExecution() {
    return context.getRemainingTimeInMillis() >= MIN_REMAINING_TIME_MILLIS;
  }

  private List<NewsRecord> getData(final LocalDate date) throws LimitExceededException, IOException, InterruptedException {
    List<NewsRecord> items = new ArrayList<>();
    try {
      JSONObject response = getJson(date, 1);
      int pages = response.getInt("pages");
      items.addAll(getItems(date, response));
      for (int page = 2; page <= pages; page++) {
        items.addAll(getItems(date, getJson(date, page)));
      }
      return items;
    } catch (JSONException e) {
      throw new LimitExceededException(e.getMessage());
    }
  }

  private List<NewsRecord> getItems(final LocalDate date, final JSONObject response) {
    List<NewsRecord> items = new ArrayList<>();
    JSONArray results = response.getJSONArray("results");
    for (int i = 0; i < results.length(); i++) {
      JSONObject result = results.getJSONObject(i);
      try {
        String pillarId = result.getString("pillarId");
        if (PILLAR_ID.equalsIgnoreCase(pillarId)) {
          String body = result.getJSONObject("fields").getString("bodyText");
          if (!body.isBlank()) {
            String id = result.getString("id");
            String title = result.getString("webTitle");
            items.add(NewsRecord.of(date, id, title, body));
          }
        }
      } catch (JSONException e) {
        continue;
      }
    }
    return items;
  }

  private JSONObject getJson(final LocalDate date, final int page) throws IOException, InterruptedException {
    HttpRequest httpRequest = HttpRequest.newBuilder(getUri(date, date, page)).build();
    waitBeforeApiCall();
    HttpResponse<String> httpResponse = handler.getHttpClient().send(httpRequest, BodyHandlers.ofString());
    String body = httpResponse.body();
    return new JSONObject(body).getJSONObject("response");
  }

  private LocalDate getStartDate(final State state) {
    LocalDate lastAddedNewsDate = state.getDate(LAST_ADDED);
    if (lastAddedNewsDate == null) {
      return AV_START_DATE;
    } else {
      return lastAddedNewsDate.plusDays(1);
    }
  }

  private URI getUri(final LocalDate from, final LocalDate to, final int page) {
    return URI.create(API_URL.formatted(handler.getApiKey(), from.toString(), to.toString(), page));
  }

  private void insert(final LocalDate date, final List<NewsRecord> records) throws Exception {
    handler.getS3Parquet().write(BUCKET_NAME, PARQUET_KEY.formatted(date), records);
  }

  private void waitBeforeApiCall() {
    long sleepMillis = lastApiCallMillis + API_CALL_MIN_FREQUENCY_MILLIS - System.currentTimeMillis();
    try {
      Thread.sleep(Duration.ofMillis(sleepMillis));
    } catch (InterruptedException e) {
      logger.log(e.getMessage(), LogLevel.ERROR);
    }
    lastApiCallMillis = System.currentTimeMillis();
  }

}
