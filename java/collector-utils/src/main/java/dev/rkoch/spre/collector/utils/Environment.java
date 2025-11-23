package dev.rkoch.spre.collector.utils;

public final class Environment {

  public static String get(final String name) {
    String value = System.getenv(name);
    if (value == null) {
      throw new IllegalStateException("%s missing in environment variables".formatted(name));
    } else {
      return value;
    }
  }

  public static long get(final String name, final long defaultValue) {
    String value = System.getenv(name);
    if (value == null) {
      return defaultValue;
    } else {
      return Long.parseLong(value);
    }
  }

  public static String get(final String name, final String defaultValue) {
    String value = System.getenv(name);
    if (value == null) {
      return defaultValue;
    } else {
      return value;
    }
  }

  private Environment() {

  }

}
