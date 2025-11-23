package dev.rkoch.spre.stock.collector.api;

public class TryAlphaVantageApi {

  public static void tryLimitExceeded() {
    AlphaVantageApi alphaVantageApi = new AlphaVantageApi();
    alphaVantageApi.getData("bf.b");
  }

  public static void main(String[] args) {
    tryLimitExceeded();
  }

}
