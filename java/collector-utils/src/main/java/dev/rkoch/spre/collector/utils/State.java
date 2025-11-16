package dev.rkoch.spre.collector.utils;

import java.nio.charset.StandardCharsets;
import java.time.LocalDate;
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

  private static final String DEFAULT_BUCKET_NAME = "dev-rkoch-spre-test";

  private static final String DEFAULT_KEY = "state.json";

  private final S3Client s3Client;

  private final String bucketName;

  private final String key;

  private final JSONObject state;

  State() {
    this(S3Client.builder().region(Region.EU_WEST_1).httpClientBuilder(UrlConnectionHttpClient.builder()).build(), DEFAULT_BUCKET_NAME, DEFAULT_KEY);
  }

  public State(S3Client s3Client, String bucketName, String key) {
    this.s3Client = s3Client;
    this.bucketName = bucketName;
    this.key = key;
    JSONObject state;
    try (ResponseInputStream<GetObjectResponse> inputStream = s3Client.getObject(GetObjectRequest.builder().bucket(bucketName).key(key).build())) {
      state = new JSONObject(new String(inputStream.readAllBytes(), StandardCharsets.UTF_8));
    } catch (NoSuchKeyException e) {
      state = new JSONObject();
    } catch (Exception e) {
      throw new RuntimeException(e.getMessage(), e);
    }
    this.state = state;
  }

  @Override
  public void close() {
    s3Client.putObject(PutObjectRequest.builder().bucket(bucketName).key(key).build(), RequestBody.fromString(state.toString()));
  }

  public String get(final String key) {
    return state.getString(key);
  }

  public LocalDate getDate(final String key) {
    return LocalDate.parse(state.getString(key));
  }

  public void set(final String key, final String value) {
    state.put(key, value);
  }

  public void setDate(final String key, final LocalDate value) {
    state.put(key, value.toString());
  }

}
