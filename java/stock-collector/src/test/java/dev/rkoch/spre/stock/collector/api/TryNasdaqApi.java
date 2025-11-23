package dev.rkoch.spre.stock.collector.api;

import java.time.LocalDate;
import dev.rkoch.spre.stock.collector.StockRecord;

public class TryNasdaqApi {

  public static void main(String[] args) throws Exception {
    tryGetData("bf.b");
    tryGetData("brk.b");
  }

  public static void tryGetData(final String symbol) throws Exception {
    NasdaqApi nasdaqApi = new NasdaqApi();
    StockRecord stockRecord = nasdaqApi.getData(LocalDate.of(2025, 11, 21), symbol);
    System.out.println(stockRecord);
  }

}
