package dev.rkoch.spre.s3.spark;

import java.time.LocalDate;

public interface DatedItem {

  String getId();

  LocalDate getLocalDate();

}
