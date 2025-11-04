package dev.rkoch.spre.s3.spark;

import java.time.LocalDate;

public class NewsItem implements DatedItem {

  public static NewsItem of(LocalDate localDate, String id, String headline, String body) {
    return new NewsItem(localDate, id, headline, body);
  }

  private final LocalDate localDate;
  private final String id;
  private final String headline;
  private final String body;

  public NewsItem(LocalDate localDate, String id, String headline, String body) {
    this.localDate = localDate;
    this.id = id;
    this.headline = headline;
    this.body = body;
  }

  public String getBody() {
    return body;
  }

  public String getHeadline() {
    return headline;
  }

  @Override
  public String getId() {
    return id;
  }

  @Override
  public LocalDate getLocalDate() {
    return localDate;
  }

}
