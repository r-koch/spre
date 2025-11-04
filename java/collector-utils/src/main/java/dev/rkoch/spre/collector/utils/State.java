package dev.rkoch.spre.collector.utils;

import java.nio.charset.StandardCharsets;
import java.time.LocalDate;
import org.json.JSONException;
import org.json.JSONObject;
import software.amazon.awssdk.core.ResponseInputStream;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.http.urlconnection.UrlConnectionHttpClient;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectResponse;
import software.amazon.awssdk.services.s3.model.NoSuchKeyException;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;

public class State implements AutoCloseable {

  private static final String DATE_LAST_ADDED_NEWS = "last.added.news.date";
  private static final String DATE_LAST_ADDED_STOCK = "last.added.stock.date";
  private static final String DATE_LIMIT_EXCEEDED_AV = "alphavantage.limit.exceeded.date";
  private static final String DATE_START_AV = "1999-11-01";
  private static final String DATE_START_NQ = "nasdaq.start.date";
  private static final String LAST_PROCESSED_STOCK = "last.processed.stock";

  private static final String DEFAULT_BUCKET_NAME = "dev-rkoch-spre-test";

  private static final String JSON_KEY = "state.json";

  private final S3Client s3Client;

  private final String bucketName;

  private final JSONObject state;

  State() {
    this(S3Client.builder().region(Region.EU_WEST_1).httpClientBuilder(UrlConnectionHttpClient.builder()).build(), DEFAULT_BUCKET_NAME);
  }

  public State(S3Client s3Client, String bucketName) {
    this.s3Client = s3Client;
    this.bucketName = bucketName;
    JSONObject state;
    try (ResponseInputStream<GetObjectResponse> inputStream = s3Client.getObject(GetObjectRequest.builder().bucket(bucketName).key(JSON_KEY).build())) {
      state = new JSONObject(new String(inputStream.readAllBytes(), StandardCharsets.UTF_8));
    } catch (NoSuchKeyException e) {
      state = new JSONObject();
    } catch (Exception e) {
      throw new RuntimeException(e.getMessage(), e);
    }
    this.state = state;
  }

  public LocalDate getLastAddedNewsDate() {
    return getDate(DATE_LAST_ADDED_NEWS);
  }

  public LocalDate getLastAddedStockDate() {
    return getDate(DATE_LAST_ADDED_STOCK);
  }

  public LocalDate getAvLimitExceededDate() {
    return getDate(DATE_LIMIT_EXCEEDED_AV);
  }

  public LocalDate getAvStartDate() {
    return LocalDate.parse(DATE_START_AV);
  }

  public LocalDate getNasdaqStartDate() {
    return getDate(DATE_START_NQ);
  }

  public String getLastProcessedStock() {
    try {
      return state.getString(LAST_PROCESSED_STOCK);
    } catch (JSONException e) {
      return null;
    }
  }

  private LocalDate getDate(final String key) {
    try {
      return LocalDate.parse(state.getString(key));
    } catch (JSONException e) {
      return null;
    }
  }

  public void setLastAddedNewsDate(final LocalDate value) {
    state.put(DATE_LAST_ADDED_NEWS, value.toString());
  }

  public void setLastAddedStockDate(final LocalDate value) {
    state.put(DATE_LAST_ADDED_STOCK, value.toString());
  }

  public void setAvLimitExceededDate(final LocalDate value) {
    state.put(DATE_LIMIT_EXCEEDED_AV, value.toString());
  }

  public void setNasdaqStartDate(final LocalDate value) {
    state.put(DATE_START_NQ, value.toString());
  }

  public void setLastProcessedStock(final String value) {
    state.put(LAST_PROCESSED_STOCK, value);
  }

  @Override
  public void close() {
    s3Client.putObject(PutObjectRequest.builder().bucket(bucketName).key(JSON_KEY).build(), RequestBody.fromString(state.toString()));
  }

}
