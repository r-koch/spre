package dev.rkoch.spre.local.collectors;

import dev.rkoch.spre.aws.utils.lambda.LocalContext;
import dev.rkoch.spre.stock.collector.Handler;
import dev.rkoch.spre.stock.collector.StockCollector;

public class LocalStockCollector {

  public static void main(String[] args) {
    new StockCollector(new LocalContext(), new Handler()).collect();
  }

}
