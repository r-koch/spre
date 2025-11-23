package dev.rkoch.spre.stock.collector.exception;

import java.time.LocalDate;

public class NoDataForDateException extends Exception {

  private static final long serialVersionUID = 1L;

  public NoDataForDateException() {
    super();
  }

  public NoDataForDateException(LocalDate date) {
    this("%s no data".formatted(date.toString()));
  }

  public NoDataForDateException(String message) {
    super(message);
  }

  public NoDataForDateException(Throwable cause) {
    super(cause);
  }

}
