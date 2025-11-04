package dev.rkoch.aws.historic.stock.collector;

import dev.rkoch.aws.stock.collector.Handler;
import dev.rkoch.aws.stock.collector.StockCollector;
import dev.rkoch.aws.utils.lambda.SystemOutLambdaLogger;

public class LocalStockCollector {

  public static void main(String[] args) {
    new StockCollector(new SystemOutLambdaLogger(), new Handler()).collect();
  }

}
