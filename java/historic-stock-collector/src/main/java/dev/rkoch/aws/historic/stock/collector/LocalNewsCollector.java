package dev.rkoch.aws.historic.stock.collector;

import java.time.LocalDate;
import dev.rkoch.aws.collector.utils.State;
import dev.rkoch.aws.news.collector.Handler;
import dev.rkoch.aws.news.collector.NewsCollector;
import dev.rkoch.aws.utils.lambda.LocalContext;
import software.amazon.awssdk.http.urlconnection.UrlConnectionHttpClient;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;

public class LocalNewsCollector {

  public static void main(String[] args) {
    // new NewsCollector(new SystemOutLambdaLogger(), new Handler()).collect();
    State state = new State(S3Client.builder().region(Region.EU_WEST_1).httpClientBuilder(UrlConnectionHttpClient.builder()).build(), "dev-rkoch-spre-test");
    LocalDate end = LocalDate.parse("1999-11-01");
    new NewsCollector(new LocalContext(), new Handler()).collect(state, end.minusDays(5), end);
  }

}
