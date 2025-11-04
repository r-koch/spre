package dev.rkoch.aws.stock.collector.exception;

import java.time.LocalDate;

public class NoDataForDateException extends Exception {

  private static final long serialVersionUID = 1L;

  public NoDataForDateException(LocalDate date) {
    super(date.toString() + " no data");
  }

  public NoDataForDateException() {
    super();
  }

  public NoDataForDateException(Throwable cause) {
    super(cause);
  }

  public NoDataForDateException(String message) {
    super(message);
  }

}
