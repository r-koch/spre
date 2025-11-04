package dev.rkoch.aws.s3.spark;

import java.time.LocalDate;

public interface DatedItem {

  String getId();

  LocalDate getLocalDate();

}
