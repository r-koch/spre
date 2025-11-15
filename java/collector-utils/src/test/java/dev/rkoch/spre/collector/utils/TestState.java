package dev.rkoch.spre.collector.utils;

import java.time.LocalDate;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class TestState {

  @Test
  public void testReadWrite() {
    try (State write = new State()) {
      write.setDate("test", LocalDate.parse("1999-12-16"));
    }

    try (State read = new State()) {
      Assertions.assertEquals(LocalDate.parse("1999-12-16"), read.getDate("test"));
    }
  }

}
