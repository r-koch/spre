package dev.rkoch.spre.stock.collector;

import java.net.http.HttpClient;
import com.amazonaws.services.lambda.runtime.Context;
import com.amazonaws.services.lambda.runtime.RequestHandler;
import dev.rkoch.spre.s3.parquet.S3Parquet;
import software.amazon.awssdk.http.urlconnection.UrlConnectionHttpClient;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;

public class Handler implements RequestHandler<Void, Void> {

  private static final String ENV_AWS_REGION = "AWS_REGION";

  private HttpClient httpClient;

  private S3Client s3Client;

  private S3Parquet s3Parquet;

  HttpClient getHttpClient() {
    if (httpClient == null) {
      httpClient = HttpClient.newHttpClient();
    }
    return httpClient;
  }

  S3Client getS3Client() {
    if (s3Client == null) {
      s3Client = S3Client.builder().region(Region.of(System.getenv(ENV_AWS_REGION))).httpClientBuilder(UrlConnectionHttpClient.builder()).build();
    }
    return s3Client;
  }

  S3Parquet getS3Parquet() {
    if (s3Parquet == null) {
      s3Parquet = new S3Parquet(getS3Client());
    }
    return s3Parquet;
  }

  @Override
  public Void handleRequest(Void input, Context context) {
    new StockCollector(context, this).collect();
    return null;
  }

}
