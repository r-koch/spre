package dev.rkoch.aws.collector.utils;

import java.time.LocalDate;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class TestState {

  @Test
  public void testReadWrite() {
    try (State write = new State()) {
      write.setLastAddedNewsDate(LocalDate.parse("1999-12-16"));
      write.setLastAddedStockDate(LocalDate.parse("2025-10-02"));
      write.setNasdaqStartDate(LocalDate.parse("2015-10-02"));
    }

    try (State read = new State()) {
      LocalDate lastAddedNewsDate = read.getLastAddedNewsDate();
      LocalDate lastAddedStockDate = read.getLastAddedStockDate();
      LocalDate nasdaqStartDate = read.getNasdaqStartDate();

      Assertions.assertEquals(LocalDate.parse("1999-12-16"), lastAddedNewsDate);
      Assertions.assertEquals(LocalDate.parse("2025-10-02"), lastAddedStockDate);
      Assertions.assertEquals(LocalDate.parse("2015-10-02"), nasdaqStartDate);
    }
  }

}
